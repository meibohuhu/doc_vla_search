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
        assistant_id=config['model']['tokens']['assistant_id']
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
    save_dir = f"runs/sft/{current_date}"

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

    trainer = pl.Trainer(
        num_nodes=1,
        max_epochs=config['training']['epochs'],
        accelerator="gpu",
        devices='auto',
        precision=trainer_precision,
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        strategy=strategy,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                dirpath=f"{save_dir}",
                filename="epoch={epoch}-loss={val_loss:.4f}",
                auto_insert_metric_name=False,
                save_weights_only=True,
                every_n_epochs=1,
            ),
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            LearningRateMonitor(logging_interval="step"),
        ],
        gradient_clip_algorithm = 'value',
        gradient_clip_val = 1.0,
######################################################### 修改这里 #########################################################
        logger=loggers,
        enable_model_summary=True,

        # limit_val_batches=0.001
    )
    torch.cuda.empty_cache()
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)