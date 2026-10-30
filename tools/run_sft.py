import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "navsim"))

import yaml
import torch
import argparse
import functools
import pytorch_lightning as pl

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import FSDPStrategy

from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import BackwardPrefetch
from torch.utils.data import DataLoader

from dataset_utils.sft_dataset import SFTDataset, DataCollator
from models.autovla import SFTAutoVLA
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
import datetime

torch.set_float32_matmul_precision('high')


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    seed_everything(args.seed)

    # Load configuration
    config = load_config(f"./config/{args.config}.yaml")

    # Model, dataset, and dataloader
    processor = AutoProcessor.from_pretrained(config['model']['pretrained_model_path'], use_fast=True)
    
    # Get using_cot setting from config (default to True if not specified)
    using_cot = config['model']['use_cot']
    

    train_dataset = SFTDataset(config['data']['train'], config['model'], processor, using_cot=using_cot)
        
    # Randomly sample from training set if train_sample_size is specified
    train_sample_size = config['training']['train_sample_size']
    if train_sample_size is not None and len(train_dataset) > train_sample_size:
        indices = torch.randperm(len(train_dataset))[:train_sample_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    else:
        print("no sampling")
        
    val_dataset = SFTDataset(config['data']['val'], config['model'], processor, using_cot=using_cot)

    model = SFTAutoVLA(config)

######################################################### 修改这里 #########################################################
    # [mh 2026/07/22] 原版只写 gradient_checkpointing_enable()，默认走 reentrant 版本，
    # 在 DDP 下会触发 reducer 的 expect_autograd_hooks_ 报错；显式指定 use_reentrant=False。
    # non-reentrant checkpointing is required for DDP static_graph (reentrant trips
    # reducer expect_autograd_hooks_) and is the recommended variant for FSDP too
    model.autovla.vlm.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    ) # enable gradient checkpointing to save memory

    # [mh 2026/07/22] 上面那句只覆盖 LLM，不含 ViT。ViT 冻结时无所谓（不留激活），
    # 但一旦解冻 ViT，每条样本 3 相机 x 4 帧 = 12 张图的激活都要留着，显存会爆，
    # 所以这里给 ViT 也单独开一次 checkpointing。
    # The LLM call above does NOT cover the vision tower. With a frozen ViT that is
    # fine (no activations retained), but once train_vision_backbone is on, the
    # activations of 12 images/sample are kept and blow up memory -- so checkpoint
    # the ViT too.
    if config['model'].get('train_vision_backbone', False):
        model.autovla.vlm.visual.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("[mem] gradient checkpointing enabled on vision tower (ViT unfrozen)")
######################################################### 修改这里 #########################################################

    # checkpoint_path = Path(".ckpt")
    # state_dict = torch.load(checkpoint_path)['state_dict']
    # model.load_state_dict(state_dict)
    
    # Create data collator with config parameters
    data_collator = DataCollator(
        processor=processor,
        ignore_index=config['model']['tokens']['ignore_index'],
        assistant_id=config['model']['tokens']['assistant_id'],
        action_start_id=config['model']['tokens']['action_start_id'],   # 解耦 cf 样本只训 action token 用
    )
    
    train_data = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        collate_fn=data_collator,
        num_workers=config['training']['num_workers'],
        shuffle=True,
    )

    val_data = DataLoader(
        val_dataset,
        batch_size=config['inference']['batch_size'],
        collate_fn=data_collator,
        num_workers=config['inference']['num_workers'],
        shuffle=False,
    )    

    # Training
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Qwen2_5_VLDecoderLayer
        },
    )

    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
######################################################### 修改这里 #########################################################
    # [mh 2026-09-04] 输出根目录改成可配置（原来写死 "runs/"，即仓库内的相对路径）。
    # 动机：3.75B 参数 fp32 的 checkpoint 约 15GB/个，config 是 save_top_k: 3
    # -> 约 45GB。而 tigris 上 /home 是 256G 配额（已用 181G，只剩 76G），
    # 放三个 checkpoint 就见底了；万一 save_weights_only 失效（连 optimizer 状态
    # 一起存就是 3 倍），直接爆配额、训练崩在最后一步。
    # 数据和模型都在 /shared（1.2T 余量），checkpoint 也该放那儿。
    #
    # 优先级：环境变量 AUTOVLA_OUTPUT_ROOT > config training.output_root > "runs"
    # 默认值保持 "runs"，所以不配置时行为和原来完全一致。
    output_root = os.environ.get("AUTOVLA_OUTPUT_ROOT",
                                 config['training'].get('output_root', 'runs'))
    save_dir = f"{output_root}/sft/{current_date}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"[output] save_dir = {os.path.abspath(save_dir)}")
######################################################### 修改这里 #########################################################

######################################################### 修改这里 #########################################################
    # [mh 2026/07/22] 原版写死 FSDP，这里改成由 config 的 training.strategy 选择（默认仍是 fsdp）。
    # 理由：FSDP FULL_SHARD 每层前向都要 all-gather 参数，只有模型单卡装不下时才划算；
    # 3B 模型单卡放得下，用 DDP（参数各卡各存一份、每步只 all-reduce 一次梯度）快得多，
    # 尤其我们这台机器没有 NVLink、NCCL 被迫走 loopback 网络传输，通信极慢。
    # Parallelism strategy is config-selectable. FSDP FULL_SHARD is only worth its
    # heavy per-layer param all-gather when the model doesn't fit; for a model that
    # fits on one GPU it is far cheaper to replicate params (DDP) and all-reduce
    # gradients once per step -- especially when the interconnect is slow (e.g. no
    # NVLink / NCCL forced onto the loopback NET transport).
    strategy_name = config['training'].get('strategy', 'fsdp')

######################################################### 修改这里 #########################################################
    # [mh 2026/07/24] fp32 master weights —— 这是一个会静默毁掉训练的坑。
    #
    # AutoVLA.__init__ 用 torch_dtype=torch.bfloat16 加载模型，而 Lightning 2.2.1 里
    # `32-true` / `bf16-mixed` 的 convert_module 都是 no-op（实测），不会把参数转回 fp32。
    # 结果：无论 DDP 还是 FSDP，AdamW 都直接在 bf16 参数上做更新。
    #
    # bf16 只有 8 位尾数（相对精度 ~0.4%），而 lr=2e-5 的单步更新相对量级约 0.1%
    # —— 低于分辨率，被反复舍入成 0。实测 200 步后：
    #     bf16 master : 只有 26% 的参数发生过变化，累积位移是 fp32 的 1/6.7
    #     fp32 master : 100% 的参数都在动
    # 症状是"模板 token 几步学完、action token 卡在 loss 3.3 三个 epoch 不动、
    # 训练集 top-1 只有 17%"——看起来像欠拟合/数据不够，其实是参数根本没在更新。
    #
    # 修法：显式把 master 转成 fp32；计算仍走 bf16（DDP 用 autocast，FSDP 用 param_dtype）。
    # 代价：单卡多约 12GB fp32 参数 + 24GB 优化器状态，80G 卡放得下。
    if config['training'].get('fp32_master', True):
        model = model.float()
        print(f"[precision] master weights -> fp32 (计算仍为 bf16)")

    _pd = next(model.parameters()).dtype
    print(f"[precision] strategy={strategy_name}  实际 param dtype={_pd}")
    if _pd != torch.float32:
        print("[precision] ⚠️  master 不是 fp32，小幅更新会被舍入，训练可能静默失效！")
######################################################### 修改这里 #########################################################

    trainer_precision = "32-true"  # FSDP 内部自己处理低精度，Trainer 这层保持 32  /  FSDP handles low precision inside the strategy
    if strategy_name == 'ddp':
        from pytorch_lightning.strategies import DDPStrategy
        # [mh 2026/07/22] find_unused_parameters=True 必须开：Qwen2.5-VL 有一部分参数不在
        # 这个 loss 的前向路径上，普通 DDP 会直接报错。static_graph 不能开——梯度累积的
        # no_sync 区间里它会触发 reducer expect_autograd_hooks_ 断言。
        # find_unused_parameters=True: some Qwen2.5-VL params are not on the forward
        # path for this loss, which plain DDP rejects. static_graph is avoided because
        # it trips reducer expect_autograd_hooks_ under grad-accumulation no_sync.
        strategy = DDPStrategy(
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
        # [mh 2026/07/24] 原为 "bf16-true"（参数/梯度/优化器状态全 bf16，无 fp32 master）
        # -> 改 "bf16-mixed"：master 保持上面转好的 fp32，前向/反向走 autocast bf16。
        # bf16-true 会把模型强制转回 bf16，等于抵消 fp32 master，务必不要用。
        # fp32_master=True  -> "bf16-mixed": master 保持 fp32，前向反向 autocast bf16
        # fp32_master=False -> "bf16-true" : 还原修复前的行为（参数/梯度/优化器全 bf16），
        #                                    仅用于做精度对照实验，正式训练不要用。
        trainer_precision = "bf16-mixed" if config['training'].get('fp32_master', True) else "bf16-true"
######################################################### 修改这里 #########################################################
    else:
        # [mh 2026/07/24] 模型已在上面转成 fp32，MixedPrecision(param_dtype=bf16) 的语义即
        # "master 保持原 dtype(fp32)、前向反向转 bf16" —— 正好是我们要的，无需额外改动。
        strategy = FSDPStrategy(
            auto_wrap_policy=wrap_policy,
            cpu_offload=False,
            # Mixed precision training
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16
            ),
            # sharding strategy
            sharding_strategy='FULL_SHARD',
            # prefetching backward computation
            backward_prefetch = BackwardPrefetch.BACKWARD_PRE,
            # save state dict type
            state_dict_type="full", # can be full or sharded
            limit_all_gathers=True, # limit all_gathers to save memory
        )

######################################################### 修改这里 #########################################################
    # [mh 2026/07/22] 加 wandb 支持，用 config 的 training.logger 开关（默认 csv）。
    # CSVLogger 永远保留，保证 runs/sft/<时间戳>/lightning_logs/.../metrics.csv 还能离线解析；
    # wandb 只是叠加上去。config=config 把整份配置一起上传，方便多次实验横向对比。
    # CSVLogger is always kept (runs/sft/<ts>/lightning_logs/.../metrics.csv stays
    # parseable); wandb is added on top when training.logger == 'wandb'.
    loggers = [CSVLogger(save_dir=f"{save_dir}")]
    if config['training'].get('logger', 'csv') == 'wandb':
        from pytorch_lightning.loggers import WandbLogger
        loggers.append(WandbLogger(
            project=config['training'].get('wandb_project', 'autovla-sft'),
            name=f"{config['name']}_{current_date}",
            save_dir=save_dir,
            config=config,          # log the full run config for comparability
        ))

    # [mh 2026-09-03] num_nodes 原本写死 1。tigris(RIT RC) 的 GH200 节点
    # **每个只有 1 张卡**(Gres=gpu:gh200:1)，所以要多卡就必须跨节点 —— 写死 1
    # 的话 Lightning 会把每个节点当成独立的单卡任务，两边各训各的、不做 all-reduce。
    # 由 config 的 training.num_nodes 指定，默认仍是 1（单机行为不变）。
    # 配合 srun + MASTER_ADDR/MASTER_PORT 使用，见 logs/0903_tiger/run_cot_sft_166k_0903_vit.sh
    # 允许用环境变量覆盖（启动脚本里设成 $SLURM_NNODES）。这样 num_nodes 永远等于
    # Slurm 实际分配的节点数，不可能和 sbatch 的 --nodes 对不上 —— 对不上时 Lightning
    # 会卡在 rendezvous 或者更糟：各节点各训各的、不做 all-reduce。
    # 没设这个环境变量时退回 config，单机行为不变。
    num_nodes = int(os.environ.get("AUTOVLA_NUM_NODES",
                                   config['training'].get('num_nodes', 1)))
    # [mh 2026-10-29] 选 ckpt / early stop 看哪个指标：training.ckpt_monitor（默认 val_loss = 原行为）。
    #   val_loss 按 token 平均，CoT / cf 训练里 val 每个 epoch 随机抽模式（reason_then_act / act_directly /
    #   cf），动作 token 占比在 epoch 间浮动（2026-10-28 那跑 21.7%–25.9%），val_loss 跟着 ±0.1 抖，
    #   和模型好坏无关。轨迹只取决于动作 token → CoT / cf 训练建议设 ckpt_monitor: val_action_loss。
    #   ckpt_mode: min（默认）/ max（如 val_action_top1）。
    #   默认 val_loss 时文件名不变（epoch=E-loss=X.ckpt）；其他指标文件名带上指标名。
    ckpt_monitor = config['training'].get('ckpt_monitor', 'val_loss')
    ckpt_mode = config['training'].get('ckpt_mode', 'min')
    if ckpt_monitor not in ('val_loss', 'val_action_loss', 'val_action_top1', 'val_text_loss'):
        raise ValueError(f"training.ckpt_monitor={ckpt_monitor} 不是 validation_step 里 log 的指标")
    ckpt_filename = ("epoch={epoch}-loss={val_loss:.4f}" if ckpt_monitor == 'val_loss'
                     else f"epoch={{epoch}}-{ckpt_monitor}={{{ckpt_monitor}:.4f}}")
    print(f"[ckpt] monitor={ckpt_monitor} mode={ckpt_mode} filename={ckpt_filename}")

    trainer = pl.Trainer(
        num_nodes=num_nodes,
        max_epochs=config['training']['epochs'],
        accelerator="gpu",
        devices='auto',
        precision=trainer_precision,
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        strategy=strategy,
        callbacks=[
            ModelCheckpoint(
                monitor=ckpt_monitor,
                mode=ckpt_mode,
                save_top_k=3,
                dirpath=f"{save_dir}",
                filename=ckpt_filename,
                auto_insert_metric_name=False,
                save_weights_only=True,
                every_n_epochs=1,
            ),
            EarlyStopping(monitor=ckpt_monitor, patience=10, mode=ckpt_mode),
            LearningRateMonitor(logging_interval="step"),
        ],
        gradient_clip_algorithm = 'value',
        gradient_clip_val = 1.0,
######################################################### 修改这里 #########################################################
        logger=loggers,
        enable_model_summary=True,
        # [mh 2026/10/27] 调试用，config 不写就和原来一样（跑满 epochs、val 全量）
        max_steps=config['training'].get('max_steps', -1),
        limit_val_batches=config['training'].get('limit_val_batches', 1.0),
        val_check_interval=config['training'].get('val_check_interval', 1.0),
    )
    torch.cuda.empty_cache()
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)