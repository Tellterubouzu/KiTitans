import argparse
import json
import random
import gzip
import numpy as np
import tqdm
import os
import shutil  # 追加：ディレクトリ削除用

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer  # トークナイザー用インポート

from adam_atan2_pytorch import AdoptAtan2
from titans_pytorch import (
    MemoryAsContextTransformer,
    MemoryMLP,
    MemoryAttention
)

import deepspeed

# wandb は local_rank==0 でのみ初期化
local_rank = int(os.environ.get("LOCAL_RANK", 0))
if local_rank == 0:
    import wandb
    WANDB_AVAILABLE = True
    PROJECT_NAME = 'titans-mac-transformer-0.3B'
    RUN_NAME = f'mac - {16} longterm mems, layers {(2,4,6,8,10,12,14)}'
    WANDB_ONLINE = True
    wandb.init(project=PROJECT_NAME, mode='disabled' if not WANDB_ONLINE else 'online')
    wandb.run.name = RUN_NAME
    wandb.run.save()
else:
    WANDB_AVAILABLE = False

# ─────────────────────────────
# ① トークナイザーの読み込み（fast tokenizer を使用）
# ─────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("weblab-GENIAC/Tanuki-8B-dpo-v1.0", use_fast=True)
NUM_TOKENS = tokenizer.vocab_size

# ─────────────────────────────
# ② ハイパーパラメータ設定
# ─────────────────────────────
NUM_BATCHES = int(55000)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 5
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 200
GENERATE_EVERY  = 500
PRIME_LENGTH = 100
GENERATE_LENGTH = 512
SHOULD_GENERATE = True
SEQ_LEN = 1024  # 固定ブロック長（シーケンス長）

# neural memory 関連など
NEURAL_MEMORY_DEPTH = 2
NUM_PERSIST_MEM = 16
NUM_LONGTERM_MEM = 16
NEURAL_MEM_LAYERS = tuple(range(2, 15, 2))
NEURAL_MEM_GATE_ATTN_OUTPUT = False
NEURAL_MEM_MOMENTUM = True
NEURAL_MEM_MOMENTUM_ORDER = 1
NEURAL_MEM_QK_NORM = True
NEURAL_MEM_MAX_LR = 1e-3
USE_MEM_ATTENTION_MODEL = False
WINDOW_SIZE = 128
NEURAL_MEM_SEGMENT_LEN = 16
NEURAL_MEM_BATCH_SIZE = 128
SLIDING_WINDOWS = True
STORE_ATTN_POOL_CHUNKS = True
MEMORY_MODEL_PER_LAYER_LEARNED_LR = True
NEURAL_MEM_WEIGHT_RESIDUAL = True
NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True

MODEL_DIM = 1024
MODEL_DEPTH = 15

# ─────────────────────────────
# ④ Lazy Loading 用の IterableDataset の定義
# ─────────────────────────────
class LazyJsonlTextDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, seq_len):
        """
        file_path: gzipped JSONL ファイルのパス
        tokenizer: transformers のトークナイザー
        seq_len: 出力する固定長ブロックの長さ
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        buffer = []
        with gzip.open(self.file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text", line)
                except json.JSONDecodeError:
                    text = line
                # 逐次トークナイズ（fast tokenizer なら encode は高速）
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                buffer.extend(token_ids)
                while len(buffer) >= self.seq_len:
                    block = buffer[:self.seq_len]
                    yield {"input_ids": torch.tensor(block, dtype=torch.long),
                           "labels": torch.tensor(block, dtype=torch.long)}
                    buffer = buffer[self.seq_len:]

# 学習データセットのパス
dataset_path = "./epoch_datasets/epoch0_dataset.jsonl.gz"
train_dataset = LazyJsonlTextDataset(dataset_path, tokenizer, SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

# 検証用：先頭から固定数ブロックを取得する関数
def get_validation_blocks(file_path, tokenizer, seq_len, max_blocks=10):
    blocks = []
    buffer = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                text = data.get("text", line)
            except json.JSONDecodeError:
                text = line
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(token_ids)
            while len(buffer) >= seq_len and len(blocks) < max_blocks:
                block = buffer[:seq_len]
                blocks.append({"input_ids": torch.tensor(block, dtype=torch.long),
                               "labels": torch.tensor(block, dtype=torch.long)})
                buffer = buffer[seq_len:]
            if len(blocks) >= max_blocks:
                break
    return blocks

val_blocks = get_validation_blocks(dataset_path, tokenizer, SEQ_LEN, max_blocks=1)
val_loader = DataLoader(val_blocks, batch_size=BATCH_SIZE)

# ─────────────────────────────
# ⑤ memory model の設定
# ─────────────────────────────
if USE_MEM_ATTENTION_MODEL:
    neural_memory_model = MemoryAttention(dim=64)
else:
    neural_memory_model = MemoryMLP(dim=64, depth=NEURAL_MEMORY_DEPTH)

# ─────────────────────────────
# ⑥ MemoryAsContextTransformer のインスタンス作成
# ─────────────────────────────
model = MemoryAsContextTransformer(
    num_tokens=NUM_TOKENS,
    dim=MODEL_DIM,
    depth=MODEL_DEPTH,
    segment_len=WINDOW_SIZE,
    num_persist_mem_tokens=NUM_PERSIST_MEM,
    num_longterm_mem_tokens=NUM_LONGTERM_MEM,
    neural_memory_layers=NEURAL_MEM_LAYERS,
    neural_memory_segment_len=NEURAL_MEM_SEGMENT_LEN,
    neural_memory_batch_size=NEURAL_MEM_BATCH_SIZE,
    neural_mem_gate_attn_output=NEURAL_MEM_GATE_ATTN_OUTPUT,
    neural_mem_weight_residual=NEURAL_MEM_WEIGHT_RESIDUAL,
    neural_memory_qkv_receives_diff_views=NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW,
    use_flex_attn=True,
    sliding_window_attn=SLIDING_WINDOWS,
    neural_memory_model=neural_memory_model,
    neural_memory_kwargs=dict(
        dim_head=64,
        heads=16,
        attn_pool_chunks=STORE_ATTN_POOL_CHUNKS,
        qk_rmsnorm=NEURAL_MEM_QK_NORM,
        momentum=NEURAL_MEM_MOMENTUM,
        momentum_order=NEURAL_MEM_MOMENTUM_ORDER,
        default_step_transform_max_lr=NEURAL_MEM_MAX_LR,
        use_accelerated_scan=True,
        per_parameter_lr_modulation=MEMORY_MODEL_PER_LAYER_LEARNED_LR
    )
).cuda()

# ─────────────────────────────
# ⑦ オプティマイザの設定
# ─────────────────────────────
optim = AdoptAtan2(model.parameters(), lr=LEARNING_RATE)

# ※ DeepSpeed を用いるため、GradScaler や torch.cuda.amp.autocast は不要です
#     （bf16 の設定は ds_config.json で管理します）

# ─────────────────────────────
# safe_decode 関数（デコード時のエラーハンドリング用）
# ─────────────────────────────
def safe_decode(token_list):
    # まず、通常の変換を試みる
    s = "".join([chr(max(32, t)) for t in token_list])
    return s.encode("utf-8", "replace").decode("utf-8")

# ─────────────────────────────
# ⑧ DeepSpeed による初期化と分散並列の設定
# ─────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # ds_config.json はコマンドラインから指定されるので、コード内には渡さない
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters()
    )

    # 学習ループ
    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=50., desc='training'):
        model_engine.train()
        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            batch = next(iter(train_loader))
            # 入力テンソルをGPUへ移動
            input_ids = batch["input_ids"].to(model_engine.device)
            loss = model_engine(input_ids, return_loss=True)
            model_engine.backward(loss)
        model_engine.step()
        if local_rank == 0:
            print(f"training loss: {loss.item()}")
            wandb.log(dict(loss=loss.item()))

        if i % VALIDATE_EVERY == 0:
            model_engine.eval()
            with torch.no_grad():
                val_batch = next(iter(val_loader))
                val_input_ids = val_batch["input_ids"].to(model_engine.device)
                val_loss = model_engine(val_input_ids, return_loss=True)
            if local_rank == 0:
                print(f"validation loss: {val_loss.item()}")
                wandb.log(dict(val_loss=val_loss.item()))

        if SHOULD_GENERATE and i % GENERATE_EVERY == 0:
            model_engine.eval()
            # 検証用データからランダムに1ブロック選択
            inp = random.choice(val_blocks)["input_ids"][:PRIME_LENGTH].to(model_engine.device, non_blocking=True)
            prime = safe_decode(inp.tolist())
            if local_rank == 0:
                print(f"{prime} \n\n {'*' * 100}")
            sample = model_engine.module.sample(inp[None, ...], GENERATE_LENGTH, use_cache=False)
            output_str = safe_decode(sample[0].tolist())
            if local_rank == 0:
                print(output_str)

        # ----- 1000毎にチェックポイント保存（最大10個まで保持） -----
        # if i % 1000 == 0 and i !=0:
        if i %1000==0:
            checkpoint_dir = "./checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            # 保存済みチェックポイント（ディレクトリ名が "step-<数字>" のもの）を取得
            existing_checkpoints = [
                d for d in os.listdir(checkpoint_dir)
                if d.startswith("step-") and os.path.isdir(os.path.join(checkpoint_dir, d))
            ]
            # 10個以上になったら、最も古いチェックポイントを削除
            if len(existing_checkpoints) >= 12:
                existing_checkpoints = sorted(existing_checkpoints, key=lambda x: int(x.split('-')[-1]))
                oldest_checkpoint = existing_checkpoints[0]
                shutil.rmtree(os.path.join(checkpoint_dir, oldest_checkpoint))
                print(f"Removed oldest checkpoint: {oldest_checkpoint}")
            model_engine.save_checkpoint(checkpoint_dir, tag=f"step-{i}")
            print(f"Saved checkpoint: step-{i}")

if __name__ == "__main__":
    main()
