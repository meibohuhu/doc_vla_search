import torch
import os
from tqdm import tqdm
from typing import Dict, Any
import pytorch_lightning as pl
from pathlib import Path
import torch.nn.functional as F
import numpy as np
from typing import List
from torch.distributed.fsdp import StateDictType
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from models.action_tokenizer import ActionTokenizer
from models.utils import rl_rewards
from dataset_utils import prompt_spec
from transformers.modeling_outputs import CausalLMOutputWithPast
from models.utils.score import PDM_Reward, TrajectorySampling, Trajectory

# # SFT 时 CoT 样本 assistant 回复的固定前缀（dataset_utils/sft_dataset.py:188）。
# # 设环境变量 AUTOVLA_FORCE_COT=1 时用它 prefill，强制模型走 slow thinking 分支。
# FORCE_COT_PREFIX = "<think>\nThis is a complex scenario requiring additional reasoning.\n"


class GRPOAutoVLA(pl.LightningModule):
    def __init__(self, config: dict, inference=False):
        super().__init__()
        self.cfg = config
        self.use_cot = config['model']['use_cot']
        self.save_hyperparameters()

        # Load trajectory sampling from config or use default
        traj_conf = config['model']['trajectory']
        self.trajectory_sampling = TrajectorySampling(
            num_poses=traj_conf['num_poses'],
            interval_length=traj_conf['interval_length']
        )
        
        # Load token configs
        token_conf = config['model']['tokens']
        self.action_start_id = token_conf['action_start_id']
        self.assistant_id = torch.tensor(token_conf['assistant_id'])

        # Training model (wrapped by Lightning FSDPStrategy)
        self.autovla = AutoVLA(config)

        self.autovla.train()
        self._train_vision_backbone = config['model']['train_vision_backbone']
        self._train_llm_backbone = config['model']['train_lm_backbone']

        # online reference model.
        if not inference:
            self.reference_model = AutoVLA(config, inference=True)
            # ★ map_location='cpu'：不加的话张量按保存设备(cuda:0)加载，DDP 下【所有 rank
            #   都堆到 cuda:0】→ 卡数一多就 OOM（6 卡 × 14GB = 84GB 爆）。载到 CPU 再由
            #   Lightning 分发到各 rank 的卡。
            state_dict = torch.load(config['model']['sft_model_path'], map_location='cpu')["state_dict"]
            state_dict = {k.replace("autovla.", "").replace("drivevla.", ""): v for k, v in state_dict.items()}
            self.reference_model.load_state_dict(state_dict, strict=False)
            self.reference_model.eval()
            # 🔴 必须冻。DDP 会给所有 requires_grad=True 的参数挂 all-reduce hook，
            #    而 reference 永远拿不到梯度 → "Expected to have finished reduction
            #    in the prior iteration" 直接报错。FSDP 下这个问题被掩盖着。
            for _p in self.reference_model.parameters():
                _p.requires_grad = False
            print(f"Using online reference model from {config['model']['sft_model_path']}")

        # ── 组采样:同一帧采 G 条 rollout ────────────────────────────────
        # 🔴 原实现的"组"是靠 tools/run_rft.py 的 GroupSampler 让【每张卡拿同一帧】、
        #    再 all_gather 凑出来的 —— 于是 G ≡ GPU 数（config devices:[0,1] → G=2，
        #    两条样本算 std 噪声极大），而且换卡数就换了算法。
        #    改成在**单卡内**采 G 条：G 与硬件解耦，各卡拿不同帧（run_rft.py 同步改回
        #    标准 DistributedSampler），DDP 的梯度平均正好是"多帧多组"。
        #    per_prompt_G = 1 时退回旧的跨卡分组，保留复现老实验的能力。
        self.per_prompt_G = int(config['rl'].get('group', {}).get('per_prompt_G', 1))

        # reward 各分项的权重（w_viol / w_say 要等 teacher 落地，见 flow §6.2/6.4/6.5）
        _rw = config['rl'].get('reward', {})
        self.w_pdm = float(_rw.get('w_pdm', 1.0))
        self.w_fmt = float(_rw.get('w_fmt', 0.0))

        # ── teacher（判决 + Δ 定价）——【默认关闭】────────────────────────
        # 打开后 reward 变成
        #     R = w_pdm·PDMS − w_fmt·1[off-format]
        #         − w_viol · 1[触发]                     ← 可行性罚
        #         − w_say  · clip(Δ,0,c) · 1[i ∈ chg]    ← 说法的标价
        # ⚠️ w_say 是**进 reward 的标价，不是 loss**。唯一的损失仍然是 L_rl。
        # 🔴 trigger 默认 nc_only:navtest 实测 69.5% 的触发是纯横向失效，
        #    而沿学生路径改速度救不了它们（见 logs/0902/step0_gonogo.md）。
        _tc = config['rl'].get('teacher', {})
        self.teacher_enable = bool(_tc.get('enable', False))
        self.w_viol = float(_tc.get('w_viol', 0.0))
        self.w_say = float(_tc.get('w_say', 0.0))
        self.clip_c = float(_tc.get('clip_c', 2.0))
        # rollout 落盘：每步把 G 条 rollout(reasoning 全文 + 轨迹 + 各分项)写 JSONL，
        # 每 rank 一个文件。供事后分析/调 reward-hacking/DiRL go-nogo（与 simlingo 的
        # rollouts.jsonl 同构）。默认开；关掉设 rl.dump_rollouts: false。
        self._dump_rollouts = bool(config['rl'].get('dump_rollouts', True)) and not inference
        self._rollout_dir = config['rl'].get('rollout_dump_dir', 'runs/grpo_rollouts')
        self._rollout_fh = None   # 懒打开（要等 global_rank 就绪）
        self._teacher = None
        self._teacher_cfg = _tc
        if self.teacher_enable and not inference:
            from models.utils.teacher import TeacherVerdict
            self._teacher = TeacherVerdict(
                Path(config['data']['train']['metric_cache_path']),
                trigger=_tc.get('trigger', 'nc_only'),
                dac_depth_min=float(_tc.get('dac_depth_min', 0.3)),
                dac_time_max=float(_tc.get('dac_time_max', 3.0)))
            print(f"[teacher] enabled  trigger={_tc.get('trigger','nc_only')}  "
                  f"w_viol={self.w_viol}  w_say={self.w_say}  clip_c={self.clip_c}")

        # sample generation config
        sample_conf = config['training']['sample']
        self._sample_generation_temperature = {
            "max_length": sample_conf['max_length'],
            "temperature": sample_conf['temperature'],
            "top_k": sample_conf['top_k'],
            "top_p": sample_conf['top_p'],
        }

        # reward function
        self.train_critic = PDM_Reward(Path(config['data']['train']['metric_cache_path']))
        self.val_critic = PDM_Reward(Path(config['data']['val']['metric_cache_path']))

        # sliding window for training reward
        if not inference:
            self.window_size = config['rl']['reward'].get("sliding_window_size", 100)
            self.register_buffer("training_reward_buffer", torch.zeros(self.window_size))
            self.register_buffer("sliding_idx",   torch.zeros(1, dtype=torch.long))
            self.register_buffer("window_count",  torch.zeros(1, dtype=torch.long))

    def training_step(self, batch):
        # Generate a sample from the model.
        self.autovla.train()
        with torch.no_grad():
            sample = self.generate_sample(
                batch, model=self.autovla, device=next(self.parameters()).device)
        
            # Compute the reward for the generated sample.
            reward = self.reward_function(sample)
            if self.teacher_enable:
                viol, delta, tinfo = self.teacher_terms(sample)
                reward = reward - self.w_viol * viol - self.w_say * delta
                self.log_dict({"t_fire": float(tinfo["fire"]) / max(len(viol), 1),
                               "t_chg": float(tinfo["chg"]) / max(len(viol), 1),
                               "t_delta": delta.mean()},
                              sync_dist=True, on_step=True, on_epoch=False)
            reward_scale = self.cfg['rl']['reward'].get("scale", 1.0)
            reward = reward * reward_scale
            # ⚠️ scale 对 advantage 无影响（组内标准化会把它约掉）；只影响 avg_train_reward
            #    这条曲线的量纲。留着是为了和旧 run 的 wandb 图可比。
            
            # Normalize the rewards to compute the advantage.
            if self.per_prompt_G > 1:
                # 组 = 同一帧的 G 条 rollout，全在本卡内。各卡拿不同帧，
                # DDP 的梯度平均 = 多帧多组，正是 GRPO 想要的。
                advantage = (reward - reward.mean()) / (reward.std() + 1e-4)
            else:
                # legacy：组 = 跨卡（GroupSampler 让每张卡拿同一帧），G ≡ GPU 数。
                # 只为复现旧实验保留，新实验请设 per_prompt_G > 1。
                groupped_rewards = self.all_gather(reward)
                advantage = (reward - groupped_rewards.mean()) / (groupped_rewards.std() + 1e-4)

        # Compute the per-token log probabilities.
        per_token_logps = self.get_per_token_logps(
            self.autovla.vlm, 
            sample['input_ids'], 
            sample['attention_mask'], 
            sample['pixel_values_videos'], 
            sample['video_grid_thw']
        )
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, sample['prompt_length']-1:]
        completion_mask = sample['completion_mask']

        # reference model
        with torch.no_grad():
            ref_per_token_logps = self.get_per_token_logps(
                self.reference_model.vlm, 
                sample["input_ids"], 
                sample["attention_mask"], 
                sample["pixel_values_videos"], 
                sample["video_grid_thw"]
            )
            ref_per_token_logps = ref_per_token_logps[:, sample["prompt_length"]-1:]

        # Compute the policy loss
        per_policy_loss = \
            torch.exp(per_token_logps - per_token_logps.detach()) * advantage.unsqueeze(-1)

        # Compute the kl loss
        kl_beta = self.cfg['rl'].get("kl_beta", 0.0)
        per_token_kl = \
            torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        per_kl_loss = kl_beta * per_token_kl

        per_token_loss = -(per_policy_loss - per_kl_loss)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log metrics
        self.log("loss", loss, sync_dist=True, prog_bar=True)
        per_kl_loss = ((per_kl_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self.log("kl_divergence", per_kl_loss, sync_dist=True)

        # record training reward
        self.training_buffer_record(reward.mean())

        # 落盘 rollout（reasoning 全文 + 轨迹 + 分项 + advantage）
        if self._dump_rollouts:
            self._dump_step_rollouts(sample, reward, advantage)
        return loss

    def _dump_step_rollouts(self, sample, reward, advantage):
        """把本步 G 条 rollout 逐条写一行 JSON 到 rank 专属文件（append）。
        字段：step / rank / token / g / text(reasoning 全文) / traj(pred waypoints) /
              pdm / format_viol / consistency / reward(含 scale) / advantage。
        任何异常都不能拖垮训练 → 整体 try/except。"""
        try:
            import json, os
            if self._rollout_fh is None:
                os.makedirs(self._rollout_dir, exist_ok=True)
                path = os.path.join(self._rollout_dir, f"rollouts_rank{self.global_rank}.jsonl")
                self._rollout_fh = open(path, "a", buffering=1)  # 行缓冲，断点也不丢
                print(f"[rollout-dump] rank{self.global_rank} → {path}")
            import numpy as np
            texts = sample.get('completion_texts', [])
            # 用 trajectory_poses（原始 (T,3) 数组，可序列化），不是 Trajectory 对象
            poses = sample.get('trajectory_poses', [])
            pdm = sample.get('_pdm'); fmt = sample.get('_fmt'); cons = sample.get('_cons')
            tok = sample.get('token')
            G = len(texts)
            for g in range(G):
                tj = poses[g] if g < len(poses) else None
                if tj is not None:
                    tj = np.asarray(tj.tolist() if hasattr(tj, 'tolist') else tj).tolist()
                rec = {
                    "step": int(self.global_step),
                    "rank": int(self.global_rank),
                    "token": tok if isinstance(tok, str) else (tok[g] if isinstance(tok, (list, tuple)) else str(tok)),
                    "g": g,
                    "text": texts[g],
                    "traj": tj,
                    "pdm": float(pdm[g]) if pdm is not None else None,
                    "format_viol": float(fmt[g]) if fmt is not None else None,
                    "consistency": float(cons[g]) if cons is not None else None,
                    "reward": float(reward[g]),
                    "advantage": float(advantage[g]),
                }
                self._rollout_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[rollout-dump] 跳过本步（不影响训练）：{e}")
    
    def training_buffer_record(self, step_reward):
        idx = self.sliding_idx.item()
        self.training_reward_buffer[idx] = step_reward

        new_idx = (idx + 1) % self.window_size
        self.sliding_idx.fill_(new_idx)
        new_count = min(self.window_count.item() + 1, self.window_size)
        self.window_count.fill_(new_count)

        if new_count >= self.window_size:
            sliding_avg = self.training_reward_buffer.mean()
            self.log(
                "avg_train_reward",
                sliding_avg,
                sync_dist=False, 
                prog_bar=True
            )

    def on_after_backward(self):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log("grad_norm", total_norm, sync_dist=True)
    
    def reward_function(self, sample):
        """R = w_pdm·PDMS − w_fmt·1[off-format]

        对 G 条 rollout 逐条算，返回 (G,) 的张量。

        🔴 一致性(说法↔轨迹)**只记录、不进 reward**(`m_consistency`)。
           把它写进 reward 等于**用手把 reasoning 和轨迹焊在一起** —— 而
           "reasoning 变好 → 轨迹变好"正是本方法要【论证】的东西(paper §7)。
           焊完再去测这条因果，测到的是自己刚加的那一项，不是方法的效果。
           所以它留在这里当**观测量**:开跑前基线是多少、DiRL 训完涨没涨，
           那才是 §7 想要的那个数。

        ⚠️ 缺的两项是 teacher 那半:
             − w_viol · 1[轨迹不可行]          需要反事实剖面 → 可行集 D（flow §6.2）
             − w_say  · clip(Δ,0,c) · 1[i∈chg] 需要 splice + Δ 定价（flow §6.4/6.5）
           它们落地前，这里就是一个"PDMS + 格式闸门 + 自洽"的普通 GRPO。

        🔴 原实现的 cot_penalty 已删:它靠 `"complex scenario" in text` 触发，
           而新 CoT 模板里这句套话不存在（实测 0/100222），一开跑就恒为 0、静默失效。
           取代它的是 w_fmt —— 直接对着新模板判格式（<think>/<answer>/<PLAN> 齐不齐、
           是否退化刷屏），见 models/utils/rl_rewards.py。
        """
        device = next(self.parameters()).device
        texts = sample['completion_texts']
        trajs = sample['trajectories']
        token = sample['token']

        pdm, fmt, cons = [], [], []
        for g, (text, traj) in enumerate(zip(texts, trajs)):
            pdm.append(float(self.train_critic.rl_pdm_score(traj, token)))
            # RL 一律走 reason_then_act（prompt 里就是这么要求的），所以 expect_think=True
            fmt.append(1.0 if rl_rewards.format_violation(text, expect_think=True) else 0.0)
            # ⚠️ 只记录，【不进 reward】—— 理由见下面 docstring
            cons.append(rl_rewards.reward_consistency(text, sample['trajectory_poses'][g]))

        pdm = torch.tensor(pdm, device=device, dtype=torch.float32)
        fmt = torch.tensor(fmt, device=device, dtype=torch.float32)
        cons = torch.tensor(cons, device=device, dtype=torch.float32)
        # 供 rollout dump 读取（不进 reward，只是把分项挂到 sample 上）
        sample['_pdm'], sample['_fmt'], sample['_cons'] = pdm, fmt, cons
        reward = self.w_pdm * pdm - self.w_fmt * fmt

        self.log_dict({
            "train_reward": reward.mean(),
            "r_pdm": pdm.mean(),
            "r_format_viol": fmt.mean(),          # 越低越好；开跑初期看它有没有下降
            "m_consistency": cons.mean(),         # ★ 指标，不是 reward。−1~1
        }, sync_dist=True, prog_bar=False, on_step=True, on_epoch=False)
        return reward

    # ────────────────────────────────────────────────────────────────────
    # teacher:判决 + Δ 定价（flow §6.4–6.5）。默认关闭。
    # ────────────────────────────────────────────────────────────────────

    def _span_nll_pair(self, text_z, span_z, text_s, span_s, sample):
        """一次前向同时算 nll(z) 和 nll(z*)，都只在各自的 [span] 上按 token 数归一。

        ⚠️ 按 span 归一，不是整句 —— 实测 SPEED 词平均 1.95 个 token，而整条 completion
           是 107 个。整句归一会把"改一个词"的信号稀释 ~55 倍，w_say 就没法定标了。

        ⚠️ 合成一个 batch 前向（而不是调两次）:每次前向都要重跑视觉编码器，
           分两次就白跑一遍。
        """
        tok = self.autovla.processor.tokenizer
        dev = sample["input_ids"].device
        pl = sample["prompt_length"]
        prompt_ids = sample["input_ids"][0, :pl]

        seqs, masks = [], []
        for text, (a, b) in ((text_z, span_z), (text_s, span_s)):
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            ids = torch.tensor(enc["input_ids"], device=dev)
            m = torch.tensor([1.0 if (o[0] < b and o[1] > a) else 0.0
                              for o in enc["offset_mapping"]], device=dev)
            if float(m.sum()) == 0:
                return None
            seqs.append(torch.cat([prompt_ids, ids]))
            masks.append(m)

        L = max(len(x) for x in seqs)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        inp = torch.full((2, L), pad, dtype=seqs[0].dtype, device=dev)
        att = torch.zeros((2, L), dtype=torch.long, device=dev)
        for i, x in enumerate(seqs):
            inp[i, :len(x)] = x
            att[i, :len(x)] = 1

        def _nll(rows):
            pv = sample["pixel_values_videos_1"].repeat(len(rows), 1)
            gt = sample["video_grid_thw_1"].repeat(len(rows), 1)
            lp = self.get_per_token_logps(self.autovla.vlm, inp[rows], att[rows], pv, gt)
            res = []
            for j, i in enumerate(rows):
                comp = lp[j, pl - 1:]
                n = min(len(comp), len(masks[i]))
                res.append(float(-(comp[:n] * masks[i][:n]).sum() / masks[i][:n].sum()))
            return res

        try:
            out = _nll([0, 1])                      # 一次前向:省一遍视觉编码
        except torch.cuda.OutOfMemoryError:
            # 峰值在视觉编码器上。显存紧时退回两次单序列 —— 慢一点，但不会崩掉训练。
            torch.cuda.empty_cache()
            out = _nll([0]) + _nll([1])
        return out[0], out[1]

    def teacher_terms(self, sample):
        """→ (viol, delta, info)。viol/delta 是 (G,) 张量;关闭时全 0。

        delta 只在 `chg`（触发 ∧ 说法也不在 D 里 ∧ 能改）上非零，其余一律 0 = 弃权。
        """
        device = next(self.parameters()).device
        G = len(sample["completion_texts"])
        viol = torch.zeros(G, device=device)
        delta = torch.zeros(G, device=device)
        info = {"fire": 0, "chg": 0, "abstain": 0}
        if self._teacher is None:
            return viol, delta, info

        from models.utils.teacher import decision_span
        for g, text in enumerate(sample["completion_texts"]):
            try:
                v = self._teacher.judge(sample["token"], sample["trajectory_poses"][g],
                                        sample["v0"], text)
            except Exception:
                continue
            if v["fire"]:
                viol[g] = 1.0
                info["fire"] += 1
            if v["target"] is None:
                info["abstain"] += int(v["fire"])
                continue
            # ⚠️ span 取【整个决策从句】("I should …: <PLAN>…</PLAN>")，不是 tag 里那个词 ——
            #    splice 同时改了散文动词和 tag，只量一处会漏掉另一处。
            sp = decision_span(text)
            sp_star = decision_span(v["z_star"])
            if sp is None or sp_star is None:
                continue
            try:
                pair = self._span_nll_pair(text, (sp[0], sp[1]),
                                           v["z_star"], (sp_star[0], sp_star[1]), sample)
            except Exception as e:
                # 🔴 不要静默吞。踩过的坑:异常被 continue 掉之后 chg 恒为 0，
                #    训练照跑，而 Δ 定价整条通路其实从没生效过。
                if not getattr(self, "_span_nll_warned", False):
                    self._span_nll_warned = True
                    print(f"[teacher] Δ 定价失败（后续不再重复报）: {type(e).__name__}: {e}",
                          flush=True)
                info["delta_error"] = info.get("delta_error", 0) + 1
                continue
            if pair is None:
                continue
            nll_z, nll_s = pair
            if not (np.isfinite(nll_z) and np.isfinite(nll_s)):
                continue
            # Δ > 0 表示 teacher 的说法比学生自己的说法【更不像】—— 这才是要付的代价。
            delta[g] = max(0.0, nll_s - nll_z)
            info["chg"] += 1
        return viol, delta.clamp(0.0, self.clip_c), info

    def get_per_token_logps(self, model, input_ids, attention_mask, pixel_values_videos, video_grid_thw):
        # Get the per-token log probabilities for the completions for the model and the reference model
        logits = model(input_ids, attention_mask=attention_mask, 
                       pixel_values_videos=pixel_values_videos, 
                       video_grid_thw=video_grid_thw).logits  # (B, L, V)
        
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it

        # 🔴 原来是 log_softmax(logits) 再 gather —— 那会把 (B, L-1, V) 整个
        #    materialize 成 float32。V=152064，B=5/L=1200 时就是 3.5 GB，
        #    纯粹为了取每个位置上一个数。
        #    等价写法:gather 之后减 logsumexp，logsumexp 在 V 维上规约掉，不留中间张量。
        #    数值上与 log_softmax 一致（logsumexp 本身就是稳定实现）。
        sel = logits.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)      # (B, L-1)
        per_token_logps = sel - torch.logsumexp(logits, dim=-1)          # (B, L-1)
        return per_token_logps

    def generate_sample(self, data, model, device):

        # Get the model inputs
        inputs = model.get_prompt(data['input_features'])
        model_inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        # 🔴 原来是 torch.manual_seed(device_index) —— 每步都把 RNG 重置成一个
        #    【与步数无关】的常数，同一张卡跨步的采样噪声因此不独立。
        torch.manual_seed((self.global_step * 8192 + self.global_rank) % (2 ** 31 - 1))

        # 同一帧采 G 条：手动把 batch 铺开 G 份再一次性 generate。
        # ⚠️ 不用 num_return_sequences —— HF 的 _expand_inputs_for_generation 对
        #    Qwen 打平的 pixel_values_videos 做 repeat_interleave 会把 patch 顺序打乱。
        #    .repeat(G, ...) 是【整块平铺】[v1v2v3 | v1v2v3 | …]，与 input_ids 里
        #    每行 3 个 video 占位符的消费顺序一致。
        G = max(1, self.per_prompt_G)
        if G > 1:
            model_inputs = {k: v.repeat(G, *([1] * (v.dim() - 1)))
                            for k, v in model_inputs.items()}

        # Generate completion
        with torch.no_grad():
            prompt_completion_ids = model.vlm.generate(
                **model_inputs,
                do_sample=True,
                max_length=self._sample_generation_temperature['max_length'],
                temperature=self._sample_generation_temperature['temperature'],
                top_k=self._sample_generation_temperature['top_k'],
                top_p=self._sample_generation_temperature['top_p'],
            )

            prompt_length = inputs.input_ids.size(1)
            prompt_mask = model_inputs['attention_mask']
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # 逐条解出轨迹（G 条）
            trajectories, trajectory_poses = [], []
            for g in range(completion_ids.size(0)):
                actions_tokens = completion_ids[g][completion_ids[g] >= self.action_start_id]
                if len(actions_tokens) > self.trajectory_sampling.num_poses:
                    actions_tokens = actions_tokens[:self.trajectory_sampling.num_poses]
                elif len(actions_tokens) < self.trajectory_sampling.num_poses:
                    actions_tokens = torch.cat([actions_tokens, torch.zeros(
                        self.trajectory_sampling.num_poses - len(actions_tokens)).to(device)]).long()
                poses = self.autovla.action_tokenizer.decode_token_ids_to_trajectory(
                    actions_tokens.cpu())[0, 1:].cpu().numpy()
                trajectory_poses.append(poses)                       # 供一致性 reward 用
                trajectories.append(Trajectory(poses, self.trajectory_sampling))

            # Create completion mask
            is_eos = completion_ids == model.processor.tokenizer.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

            # Concatenate prompt_mask with completion_mask for logit computation
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1) 

            completion_texts = model.processor.batch_decode(completion_ids)

            # Create outputs
            # 单序列前向（teacher 的 Δ 定价）要用【未平铺】的一份 video 特征
            _np = model_inputs['pixel_values_videos'].shape[0] // max(1, G)
            _ng = model_inputs['video_grid_thw'].shape[0] // max(1, G)
            _v = data['input_features'].get('vehicle_velocity', [0.0, 0.0])
            _v0 = float(np.hypot(float(_v[0]), float(_v[1]))) if len(_v) >= 2 else float(_v[0])
            outputs = {'trajectories': trajectories,
                       'trajectory_poses': trajectory_poses,
                       'v0': _v0,
                       'pixel_values_videos_1': model_inputs['pixel_values_videos'][:_np],
                       'video_grid_thw_1': model_inputs['video_grid_thw'][:_ng],
                       'token': data['token'], 
                       'completion_texts': completion_texts,
                       'prompt_length': prompt_length,
                       'input_ids': prompt_completion_ids, 
                       'completion_ids': completion_ids,
                       'attention_mask': attention_mask,
                       'completion_mask': completion_mask,
                       'pixel_values_videos': model_inputs['pixel_values_videos'], 
                       'video_grid_thw': model_inputs['video_grid_thw'],
                        }
        
        # clean up
        torch.cuda.empty_cache()

        return outputs
    
    def configure_optimizers(self):
        # LoRA 模式：requires_grad 已在 __init__ 里由 peft + ViT 重开设好，跳过 flag 冻结。
        # peft 包装后模块路径也变了（vlm.visual -> vlm.base_model.model.visual），
        # 走原来的 flag 分支会 AttributeError。
        if not getattr(self, '_is_lora', False):
            if not self._train_vision_backbone:
                for param in self.autovla.vlm.visual.parameters():
                    param.requires_grad = False

            if not self._train_llm_backbone:
                for param in self.autovla.vlm.model.parameters():
                    param.requires_grad = False

        params_to_update = []
        for param in self.autovla.vlm.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

        assert len(params_to_update) > 0, 'No parameters to update'

        lr = float(self.cfg['training']['learning_rate'])
        wd = float(self.cfg['training'].get('weight_decay', 0.0))
        optimizer = torch.optim.AdamW(
            params_to_update,
            lr=lr,
            weight_decay=wd
        )

        return optimizer
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        # Filter out parameters with no gradient to avoid empty tensor lists
        params_with_grad = [p for p in self.parameters() if p.grad is not None]
        if params_with_grad:
            torch.nn.utils.clip_grad_value_(params_with_grad, clip_value=gradient_clip_val)

    def on_save_checkpoint(self, checkpoint: dict):
        # only save main model
        sd = checkpoint.get("state_dict", {})
        for k in list(sd):
            if k.startswith("reference_model."):
                sd.pop(k)

class SFTAutoVLA(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config
        self.save_hyperparameters()

        self.autovla = AutoVLA(config)
        self.autovla.train()

        self._train_vision_backbone = config['model']['train_vision_backbone']
        self._train_llm_backbone = config['model']['train_lm_backbone']

######################################################### 修改这里 #########################################################
        # [mh 2026/07/25] LLM LoRA 模式。train_lm_backbone 现在是三态：
        #   True   -> LLM 全参
        #   False  -> LLM 冻结
        #   "lora" -> LLM 走 LoRA（本分支）
        # 配合 train_vision_backbone(true/false) 可组合出 4 种：
        #   ViT全参+LLM全参 / ViT全参+LLM_LoRA / ViT冻+LLM全参 / ViT冻+LLM_LoRA
        #
        # ⚠️ modules_to_save 必须含 embed_tokens/lm_head：模型要学 2048 个新的
        # <action_i> token（resize 后是随机初始化），而 LoRA 只作用于 target_modules
        # (q/k/v/o_proj)，碰不到 embedding。不 save embedding 的话 action token 永远
        # 学不会，PDMS 会很低——症状酷似之前的 bf16 bug，极易误判。
        # Qwen2.5-VL-3B 是 tie_word_embeddings=True，两个都列由 peft 处理绑定。
        self._is_lora = (str(self._train_llm_backbone).lower() == 'lora')
        if self._is_lora:
            from peft import LoraConfig, get_peft_model, TaskType
            lc = config['model'].get('lora', {})
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=lc.get('target_modules', ['q_proj', 'v_proj', 'k_proj', 'o_proj']),
                modules_to_save=lc.get('modules_to_save', ['lm_head', 'embed_tokens']),
                r=lc.get('r', 16),
                lora_alpha=lc.get('alpha', 32),
                lora_dropout=lc.get('dropout', 0.05),
                bias='none',
            )
            self.autovla.vlm = get_peft_model(self.autovla.vlm, lora_config)
            # get_peft_model 会冻结【所有】非 LoRA / 非 modules_to_save 的参数——包括 ViT。
            # 所以想训 ViT 必须在这里手动重开，否则 config 写 true 也没用。
            if self._train_vision_backbone:
                for p in self.autovla.vlm.base_model.model.visual.parameters():
                    p.requires_grad = True
                print("[lora] ViT unfrozen (full-param) on top of LLM-LoRA")
            self.autovla.vlm.print_trainable_parameters()
######################################################### 修改这里 #########################################################

    def training_step(self, batch):
        hascot = batch['has_cot']
        gt_trajectory = batch["gt_trajectory"]
        gt_action = batch["gt_action"]
        output = self.autovla(batch)
        loss = output.loss

        # === Add additional loss on action tokens ===
        # output.logits shape: (B, T, V), labels shape: (B, T)
        logits = output.logits
        vocab_size = logits.size(-1)
        # Flatten logits and labels for token-wise loss
        labels = batch['labels']
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        logits_flat = shift_logits.view(-1, vocab_size)
        labels_flat = shift_labels.view(-1)
        # Identify action token positions
        action_mask = (labels_flat >= self.autovla.action_start_id)  # shape: (B*T,)
        # Compute token-wise cross-entropy loss
        ce_loss_all = F.cross_entropy(logits_flat, labels_flat, reduction='none')  # shape: (B*T,)
        # Extract loss for action tokens
        action_loss = ce_loss_all[action_mask]
        # Add to total loss with optional weighting factor
        if action_loss.numel() > 0:
            action_loss = action_loss.mean()

######################################################### 修改这里 #########################################################
        # [mh 2026/07/24] 把 action token 的 loss / top-1 单独记录出来。
        # 原版只 log 总 loss，而总 loss 里约 69% 是早已学到 ~0 的固定模板 token
        # （"<answer>The final output action is:" 这些），只有 ~31% 是真正要学的
        # 10 个 <action_i>。结果就是 val_loss=1.05 看着"在收敛"，实际 action loss
        # 还有 3.3（困惑度 27），模型根本没学会开车——这个假象曾让我们误判为
        # 数据不足/欠拟合，实际是 bf16 master 导致参数没更新（见 run_sft.py 的精度注释）。
        # 教训：永远盯真正要学的那部分 loss，不要盯被模板稀释过的平均值。
        with torch.no_grad():
            if action_mask.any():
                _al = ce_loss_all[action_mask].mean()
                _acc = (logits_flat[action_mask].argmax(-1) == labels_flat[action_mask]).float().mean()
                self.log("train_action_loss", _al, sync_dist=True, prog_bar=True,
                         batch_size=gt_action.shape[0])
                self.log("train_action_top1", _acc, sync_dist=True, prog_bar=True,
                         batch_size=gt_action.shape[0])
            # [mh 2026/09/02] 同时盯【非 action 的 target token】的 loss。
            # 开 CoT 后这部分主要就是 reasoning(37 tok) + 模板(43 tok)，
            # 模板很快到 ~0，所以它基本反映 reasoning 学得怎么样。
            # 为什么要单独看:加了 L_action 之后 action 拿走了绝大部分梯度
            #   (按基线实测的 CE 量级估,初期 ~96%、收敛后 ~92%),
            #   reasoning 只剩个位数百分比。reasoning 到底有没有被学会
            #   不能靠猜，只能看这条曲线 + 采样生成。
            # ⚠️ 只在 has_cot 的 step 上记，否则 act_directly 档的纯模板 loss 会把它拉平。
            # 🔴 这里【不能】按 has_cot 分支来决定记不记。
            #    sync_dist=True 会触发一次跨 rank 的 all-reduce，而 has_cot 是逐样本
            #    随机的(75/25)，4 个 rank 极少一致 —— 有的 rank 进集合通信、有的不进，
            #    直接 DDP 死锁(GPU 占用 100% 但一步都不走)。2026-09-02 就是这么挂的。
            #    所以无条件记:所有 rank 走同一条路径。
            # ⚠️ 代价:act_directly 档这里只有模板 token(loss 很快 ~0)，会把曲线拉低。
            #    但 75% 的 step 是 CoT，模板项趋近 0，曲线仍然主要反映 reasoning。
            _tm = (labels_flat != -100) & (~action_mask)
            if _tm.any():
                self.log("train_text_loss", ce_loss_all[_tm].mean(), sync_dist=True,
                         batch_size=gt_action.shape[0])
######################################################### 修改这里 #########################################################

######################################################### 修改这里 #########################################################
        # [mh 2026/09/02] 原上游代码:
        #     if hascot[0] == True:
        #         loss = loss * 40          # = 论文里的 lambda_cot
        #         loss = loss + action_loss # = 论文里的 lambda_a * L_action
        # 没有 else 分支 —— 即 no-CoT 样本既不乘 40 也不加 action_loss。
        #
        # 作者在 issue #23 里解释了这两项的用意:
        #   · L_action 是为了对冲【reasoning 把 action token 稀释掉】——
        #     "action tokens are heavily diluted by the much longer reasoning text"。
        #   · x40 (lambda_cot) 是因为【CoT 样本比 action-only 样本稀少得多】,
        #     "we apply a larger weight to boost the reasoning loss and balance
        #      the joint training"。
        #
        # 🔴 第二个前提在我们这里是【反的】:cot_ratio=0.75，CoT 样本占多数。
        #    照搬 x40 等于把多数档的梯度放大 40 倍、少数档留在 1 倍，
        #    并且 no-CoT 基线(use_cot:false ⇒ has_cot 恒 False)从来没走过这条路，
        #    CoT/no-CoT 的对比会被一个 40 倍的梯度尺度差污染。
        #    同一个 issue 里两位使用者报告的正是这个后果:SFT 完模型只吐 CoT 文本、
        #    一个 action token 都不生成 —— 被放大的 L_LM 由 reasoning 文本主导。
        #
        # 改成作者给的通式 L = w_i * L_LM + lambda_a * L_action，取 w_i = lambda_a = 1，
        # 两档同等对待:
        #   · 去掉 x40    —— 频率前提不成立
        #   · 两档都加 action_loss —— 稀释是真的(实测 action token 在
        #     reason_then_act 档只占 target 的 11%，act_directly 档 19%，差 1.69x),
        #     而且论文公式本来就对所有样本加这一项，上游代码漏了 no-CoT 分支
        #     (issue #23 提问者指出的就是这个)。
        #
        # ⚠️ 与已跑完的 no-CoT 基线相比，这里多了 action_loss 一项。
        #    要做严格可比的 CoT vs no-CoT，基线得用同一个 loss 重跑。
        if action_loss.numel() > 0:
            loss = loss + action_loss
######################################################### 修改这里 #########################################################

        self.log("train_loss", loss.item(),
                 batch_size=gt_action.shape[0],
                 sync_dist=True,
                 prog_bar=True)
        
        
        return loss
    
    def validation_step(self, batch):
        gt_trajectory = batch["gt_trajectory"]
        gt_action = batch["gt_action"]
        # ⚠️ 必须在 self.autovla(batch) 之前取 —— forward 里会 inputs.pop('has_cot')，
        #    调用之后再读就是 KeyError(training_step 里也是先取后调用)。
        hascot = batch['has_cot']

        output = self.autovla(batch)
        loss = output.loss
        self.log("val_loss", loss.item(),
                 batch_size=gt_action.shape[0],
                 sync_dist=True, prog_bar=True)

######################################################### 修改这里 #########################################################
        # [mh 2026/07/24] 同 training_step：val_loss 被模板 token 稀释，
        # 选 checkpoint / 判断是否收敛都应该看 val_action_loss 和 val_action_top1。
        with torch.no_grad():
            _lg = output.logits[..., :-1, :].contiguous().view(-1, output.logits.size(-1))
            _lb = batch['labels'][..., 1:].contiguous().view(-1)
            _m = _lb >= self.autovla.action_start_id
            if _m.any():
                _al = F.cross_entropy(_lg[_m].float(), _lb[_m])
                _acc = (_lg[_m].argmax(-1) == _lb[_m]).float().mean()
                self.log("val_action_loss", _al, sync_dist=True, prog_bar=True,
                         batch_size=gt_action.shape[0])
                self.log("val_action_top1", _acc, sync_dist=True, prog_bar=True,
                         batch_size=gt_action.shape[0])
            # 同 training_step:不能按 has_cot 分支，否则 sync_dist 的 all-reduce 会错配死锁
            _tm = (_lb != -100) & (~_m)
            if _tm.any():
                self.log("val_text_loss", F.cross_entropy(_lg[_tm].float(), _lb[_tm]),
                         sync_dist=True, batch_size=gt_action.shape[0])
######################################################### 修改这里 #########################################################
        
        return loss
    
    def configure_optimizers(self):
        if not self._train_vision_backbone:
            for param in self.autovla.vlm.visual.parameters():
                param.requires_grad = False

        if not self._train_llm_backbone:
            for param in self.autovla.vlm.model.parameters():
                param.requires_grad = False

######################################################### 修改这里 #########################################################
        lr = float(self.cfg['training']['learning_rate'])
        # [mh 2026/07/22] 原版所有可训练参数共用一个 lr。这里把 ViT 拆成独立的参数组：
        # 主 lr（2e-5）是按 LLM 调的，对一个预训练好的 ViT 来说太大——解冻 ViT 时共用这个 lr，
        # 几百步就能把预训练的视觉特征训坏。config 不写 vision_learning_rate 时它等于 lr，
        # 所以行为和原版完全一致（ViT 冻结的 baseline 不受影响）。
        # Separate param group for the vision tower. The main lr (2e-5) is tuned for
        # the LLM and is too large for a pretrained ViT -- when train_vision_backbone
        # is on, sharing it can wreck the pretrained features within a few hundred
        # steps. Defaults to `lr`, so behaviour is unchanged when left unset.
        vision_lr = float(self.cfg['training'].get('vision_learning_rate', lr))

        # [mh 2026/07/22] 按 id() 区分 ViT 参数和其余参数（不能按名字，这里拿到的是裸 Parameter）
        vision_param_ids = {id(p) for p in self.autovla.vlm.visual.parameters()}
        vision_params, other_params = [], []
        for param in self.autovla.vlm.parameters():
            if not param.requires_grad:
                continue
            (vision_params if id(param) in vision_param_ids else other_params).append(param)

        assert vision_params or other_params, 'No parameters to update'

        # [mh 2026/07/22] 两个参数组分别带自己的 lr；ViT 全冻结时 vision_params 为空，
        # 就退化成原来的单组写法。下面 print 出来是为了在日志里核对到底解冻了多少参数。
        param_groups = []
        if other_params:
            param_groups.append({'params': other_params, 'lr': lr})
        if vision_params:
            param_groups.append({'params': vision_params, 'lr': vision_lr})
            print(f"[optim] vision tower trainable: {sum(p.numel() for p in vision_params)/1e6:.0f}M "
                  f"params @ lr={vision_lr:g} | rest: "
                  f"{sum(p.numel() for p in other_params)/1e6:.0f}M @ lr={lr:g}")

######################################################### 修改这里 #########################################################
        # [mh 2026/09/04] 优化器可选，用来把解冻 ViT 后的显存压进 80G 卡。
        #
        # 背景：3.8B 全参 + fp32 master 时静态显存 = 参数14 + 梯度14 + AdamW(m,v)28 = 56GB，
        # 加激活/碎片实测峰值 84.6GB，塞不进 80G 卡。bs 已是 1、LLM 和 ViT 的
        # gradient checkpointing 都已开，唯一还能砍的大头就是那 28GB 优化器状态。
        #
        # 本机 GPU0 实测（3.76B 可训练参数，真实 step 后量的）：
        #   adamw    : 优化器状态 28.0 GB  -> 静态 56.0 GB
        #   adamw8bit: 优化器状态  7.1 GB  -> 静态 35.1 GB   省 20.9 GB
        #
        # 'adamw'(默认) 保持原行为，老实验完全不受影响；换 8bit 才需在 config 里显式写。
        #   adamw8bit    : m/v 量化到 8bit，省 21GB。LLM 微调的标准做法，实践中与 fp32
        #                  几乎无差别，但【不是逐比特一致】。
        #   pagedadamw32 : 状态仍是 fp32（数学完全一致），显存吃紧时自动分页到 CPU。
        #                  要逐比特复现就用它，代价是分页时变慢。
        opt_name = str(self.cfg['training'].get('optimizer', 'adamw')).lower()
        wd = float(self.cfg['training'].get('weight_decay', 0.0))
        if opt_name == 'adamw':
            optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)
        else:
            try:
                import bitsandbytes as bnb
            except ImportError as e:
                raise ImportError(
                    f"training.optimizer={opt_name} 需要 bitsandbytes：pip install bitsandbytes"
                ) from e
            cls = {'adamw8bit': bnb.optim.AdamW8bit,
                   'pagedadamw32': bnb.optim.PagedAdamW32bit}.get(opt_name)
            if cls is None:
                raise ValueError(
                    f"未知 training.optimizer={opt_name}；可选 adamw / adamw8bit / pagedadamw32")
            optimizer = cls(param_groups, lr=lr, weight_decay=wd)
        print(f"[optim] optimizer={opt_name}")
######################################################### 修改这里 #########################################################
        lr_warmpup_step = self.cfg['training']['lr_warmup_step']
        lr_step_freq = self.cfg['training']['lr_step_frequency']
        lr_step_gamma = self.cfg['training']['lr_step_gamma']

        def lr_update(step, warmup_step, step_size, gamma):
            if step < warmup_step:
                # warm up lr
                lr_scale = 1 - (warmup_step - step) / warmup_step * 0.95
            else:
                n = (step - warmup_step) // step_size
                lr_scale = gamma ** n

            if lr_scale < 1e-2:
                lr_scale = 1e-2
            elif lr_scale > 1:
                lr_scale = 1

            return lr_scale
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: lr_update(
                step,
                lr_warmpup_step,
                lr_step_freq,
                lr_step_gamma,
            )
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    @torch.no_grad()
    def calculate_metrics(self, logits, labels, gt_trajectory):
        # Find start index for ground truth sequence
        gt_start_idx = self.find_assistant_start_idx(labels[0])
        gt_tokens = labels[0, gt_start_idx+1:] # shifted
        pred_tokens = logits[0, gt_start_idx:-1].argmax(dim=-1)

        # Find action tokens in ground truth and predicted sequences
        gt_action_idx = gt_tokens >= self.autovla.action_start_id
        pred_action_idx = pred_tokens >= self.autovla.action_start_id

        if len(pred_tokens[pred_action_idx]) != len(gt_tokens[gt_action_idx]):
            pred_action_idx = gt_action_idx
            
        gt_action_tokens = gt_tokens[gt_action_idx]
        pred_action_tokens = pred_tokens[pred_action_idx]

        # Decode predicted trajectory
        # pred_trajectory = self.autovla.action_tokenizer.decode_token_ids_to_trajectory(pred_action_tokens.cpu())
        # action_acc = (pred_action_tokens == gt_action_tokens).float().mean()
        # traj_mse = torch.norm(pred_trajectory[0, 1:, :2] - gt_trajectory[0].cpu(), dim=-1).mean()
        # traj_mse = traj_mse.to(logits.device)

        # return {
        #     'action_acc': action_acc,
        #     'traj_mse': traj_mse
        # }
    
    @staticmethod
    def find_assistant_start_idx(labels):
        assistant_id = torch.tensor(ASSISTANT_ID).to(labels.device)
        
        for j in range(len(labels) - len(assistant_id) + 1):
            if torch.equal(labels[j:j + len(assistant_id)], assistant_id):
                start_idx = j
                break

        return start_idx

######################################################### 修改这里 #########################################################
def load_vlm(model_path, device):
    """Load the VLM backbone.

    [mh 2026/07/22] 单独抽出来做一个统一的加载入口，行为和原版写死的
    Qwen2_5_VLForConditionalGeneration.from_pretrained 完全一致。
    之前这里有一段按 checkpoint 的 model_type 自动切 Qwen3-VL 模型类的逻辑，
    现在先删掉（Qwen3-VL 需要 transformers>=4.57，当前环境用不上）；
    以后要换 backbone 只改这一个函数即可。
    """
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    )
######################################################### 修改这里 #########################################################

class AutoVLA(torch.nn.Module):
    def __init__(self, config, inference=False, device='cpu'):
        super().__init__()
        self.device = device

        model_path = config['model']['pretrained_model_path']
        self.vlm = load_vlm(model_path, device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer, 
                                                model_config=config['model'])
        self.vlm.resize_token_embeddings(len(self.processor.tokenizer))

        self.video_conf = config['model']['video']
        self.action_start_id = config['model']['tokens']['action_start_id']

        self.use_cot = config['model']['use_cot']
        self.gen_conf = config['inference']['sample']

    def predict(self, input_features):
        inputs = self.get_prompt(input_features)
        model_inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        # --- mh 26-07-23: 噪声对照实验用的采样覆盖 ---
        # 默认 temperature=0.01（近乎贪心，每次吐同一串 action token）。
        # 设 AUTOVLA_SAMPLE_TEMP/TOP_P 可覆盖，用于检验"CoT 救回归零场景"到底是
        # 推理起作用，还是仅仅换了个采样把越界的 token 换掉了（对照组：不开 CoT，只加温度）。
        _temp = float(os.environ.get("AUTOVLA_SAMPLE_TEMP", self.gen_conf['temperature']))
        _top_p = float(os.environ.get("AUTOVLA_SAMPLE_TOP_P", self.gen_conf['top_p']))
        _top_k = int(os.environ.get("AUTOVLA_SAMPLE_TOP_K", self.gen_conf['top_k']))
        outputs = self.vlm.generate(
            **model_inputs,
            max_length=self.gen_conf['max_length'],
            do_sample=True,
            temperature=_temp,
            top_k=_top_k,
            top_p=_top_p,
        )

        outputs_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]

        outputs_trimmed = outputs_trimmed[0][:-1].cpu() # remove end token
        cot_results = self.processor.decode(outputs_trimmed)
        # # prefill 的前缀属于输入、不在 generate 的输出里，补回来才是完整回复 mh 2026/07/25 测试用的
        # if os.environ.get("AUTOVLA_FORCE_COT") and self.use_cot:
        #     cot_results = FORCE_COT_PREFIX + cot_results

        # if 'Chain-of-Thought is not needed' not in self.processor.decode(outputs_trimmed):
        #     print(self.processor.decode(outputs_trimmed))
        #     print("has cot")
        # else:
        #     print(self.processor.decode(outputs_trimmed))
        #     print("no cot")
        actions_tokens = outputs_trimmed[outputs_trimmed >= self.action_start_id]

        trajectory = self.action_tokenizer.decode_token_ids_to_trajectory(actions_tokens)[0, 1:]

        return trajectory, cot_results
    
    def get_prompt(self, input_features, image_mode="video"):
        # image sensor
        images = input_features['images']

        min_pixels = self.video_conf.get("min_pixels", 28 * 28 * 128)
        max_pixels = self.video_conf.get("max_pixels", 28 * 28 * 128)

        camera_images = {}
        
        # List of camera types to load
        camera_types = ['front_camera', 'front_left_camera', 'front_right_camera']
        
        # When sensor_data_path is set, image paths are relative and need the prefix.
        # When it is null/empty (e.g. nuScenes stores full paths), use them as-is.
        for camera_type in camera_types:
            camera_images[camera_type] = []
            for i in range(4):
                img = images[camera_type][i]
                if input_features.get('sensor_data_path'):
                    camera_images[camera_type].append(
                        os.path.join(input_features['sensor_data_path'], img))
                else:
                    camera_images[camera_type].append(img)

        # Assign to individual variables for message formatting
        front_camera_1, front_camera_2, front_camera_3, front_camera_4 = camera_images['front_camera']
        front_left_camera_1, front_left_camera_2, front_left_camera_3, front_left_camera_4 = camera_images['front_left_camera']
        front_right_camera_1, front_right_camera_2, front_right_camera_3, front_right_camera_4 = camera_images['front_right_camera']


        # vehicle state
        velocity = input_features["vehicle_velocity"]

        if isinstance(velocity, list) or isinstance(velocity, np.ndarray):
            velocity_x = velocity[0]
            velocity_y = velocity[1]
            velocity = np.sqrt(velocity_x**2 + velocity_y**2)
    
        acceleration = input_features["vehicle_acceleration"]
        if isinstance(acceleration, list) or isinstance(acceleration, np.ndarray):
            acceleration_x = acceleration[0]
            acceleration_y = acceleration[1]
            acceleration = np.sqrt(acceleration_x**2 + acceleration_y**2)

        instruction = input_features["driving_command"].lower()
    
        user_content = [
            {
                "type": "text",
                "text": (
                    "The autonomous vehicle is equipped with three cameras mounted at the front, left, and right, enabling a comprehensive perception of the surrounding environment."
                )
            },
            {
                "type": "text",
                "text": "The first video presents the front view of the vehicle, comprising four sequential frames sampled at 2 Hz."
            },
            {
                "type": "video",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "video": [
                    f"file://{front_camera_1}",
                    f"file://{front_camera_2}",
                    f"file://{front_camera_3}",
                    f"file://{front_camera_4}",
                ]
            },
            {
                "type": "text",
                "text": "The second video presents the front-left view of the vehicle, comprising four sequential frames sampled at 2 Hz."
            },
            {
                "type": "video",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "video": [
                    f"file://{front_left_camera_1}",
                    f"file://{front_left_camera_2}",
                    f"file://{front_left_camera_3}",
                    f"file://{front_left_camera_4}",
                ]
            },
            {
                "type": "text",
                "text": "The third video presents the front-right view of the vehicle, comprising four sequential frames sampled at 2 Hz."
            },
            {
                "type": "video",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "video": [
                    f"file://{front_right_camera_1}",
                    f"file://{front_right_camera_2}",
                    f"file://{front_right_camera_3}",
                    f"file://{front_right_camera_4}",
                ]
            },
            {
                "type": "text",
                # 🔴 与 SFT 共用同一份文案（dataset_utils/prompt_spec.py）。
                #    以前这里是独立的一份、且已经漂到旧版 —— RL rollout 于是跑在
                #    训练分布外。RL 一律走 reason_then_act:方法要改写的就是那句
                #    <PLAN>，模型必须先把它说出来。
                "text": prompt_spec.user_text(
                    velocity, acceleration, instruction, self.use_cot,
                    cot_mode="reason_then_act"),
            },
        ]

        messages = [
            {"role": "system",
             "content": [{"type": "text", "text": prompt_spec.system_text(self.use_cot)}]},
            {"role": "user", "content": user_content},
        ]

        image_inputs, video_inputs = process_vision_info(messages)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )

        # # --- mh 26-07-22: 强制开启 CoT（消融用）---
        # # 发布的 RFT ckpt 在 navtest 上 100% 走 fast thinking：<think> 里恒为
        # # "This is a straightforward scenario..."，一次真实推理都不做。
        # # 想量化 reasoning 到底有没有用，就得强制它进 slow 分支。
        # #
        # # 做法：prefill assistant 回复的开头。SFT 时 CoT 样本的前缀就是下面这句
        # # (dataset_utils/sft_dataset.py:188)，所以这是**完全在训练分布内**的干预，
        # # 比改 system prompt 让它"必须推理"要干净得多。
        # if os.environ.get("AUTOVLA_FORCE_COT") and self.use_cot:
        #     text = text + FORCE_COT_PREFIX

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        return inputs
    
    def forward(self, inputs):
        inputs.pop('gt_trajectory')
        inputs.pop('gt_action')
        inputs.pop('has_cot')
        outputs: CausalLMOutputWithPast = self.vlm(**inputs)

        return outputs