import argparse
import json
import random
import numpy as np
import tqdm
import os
import datetime
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_DISTRIBUTED_DEFAULT_TIMEOUT"] = str(datetime.timedelta(hours=1))

import shutil
import math
from itertools import chain, islice

import torch
from torch.utils.data import DataLoader, IterableDataset

# huggingface datasets
from datasets import load_dataset

# AutoTokenizer に加え、cosineスケジューラー用の関数をインポート
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from adam_atan2_pytorch import AdoptAtan2
from titans_pytorch import (
    MemoryAsContextTransformer,
    MemoryMLP,
    MemoryAttention
)

import deepspeed
local_rank = int(os.environ.get("LOCAL_RANK", 0))

# ─────────────────────────────
# ① トークナイザーの読み込み（fast tokenizer を使用）
# ─────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("weblab-GENIAC/Tanuki-8B-dpo-v1.0", use_fast=True)
NUM_TOKENS = tokenizer.vocab_size

# ─────────────────────────────
# ② ハイパーパラメータ設定
# ─────────────────────────────
# ※以降のエポックで使用するシーケンス長はエポックごとに変更するので、ここではもともとの値は参考程度
BATCH_SIZE = 5  # ※後ほど各エポックで上書きします
GRADIENT_ACCUMULATE_EVERY = 5
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 200
GENERATE_EVERY  = 500
PRIME_LENGTH = 100
GENERATE_LENGTH = 512
SHOULD_GENERATE = True
SEQ_LEN = 1024

# neural memory 関連
NEURAL_MEMORY_DEPTH = 2
NUM_PERSIST_MEM = 16
NUM_LONGTERM_MEM = 16
NEURAL_MEM_LAYERS = tuple(range(2, 15, 2))
NEURAL_MEM_GATE_ATTENTION_OUTPUT = False
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

NUM_BATCHES = 25000

# ─────────────────────────────
# ③ 各データセット（streaming版）のロードと変換処理
# ─────────────────────────────

# SyntheticTextWikiTranslate： flat_map_iterable を用いて、各サンプルから "ja" と "eng" を個別サンプルに展開
def flat_map_iterable(dataset, fn):
    for sample in dataset:
        for item in fn(sample):
            yield item

raw_wiki_stream = load_dataset(
    "kanhatakeyama/SyntheticTextWikiTranslate",
    split="train",
    streaming=True,
).shuffle(buffer_size=100)
#dataset_wiki_stream = flat_map_iterable(raw_wiki_stream, lambda sample: [{"text": sample.get("ja", "")}, {"text": sample.get("eng", "")}])
dataset_wiki_stream = flat_map_iterable(raw_wiki_stream, lambda sample: [{"text": sample.get("ja", "")}])

# SyntheticTextCC： "ja" 列を使用（シャッフル）
raw_cc_stream = load_dataset(
    "kanhatakeyama/SyntheticTextCC",
    split="train",
    streaming=True,
).shuffle(buffer_size=100)
#dataset_cc_stream = flat_map_iterable(raw_cc_stream, lambda sample: [{"text": sample.get("ja", "")}, {"text": sample.get("eng", "")}])
dataset_cc_stream = flat_map_iterable(raw_wiki_stream, lambda sample: [{"text": sample.get("ja", "")}])

# ─────────────────────────────
# ③-2 カスタム IterableDataset の定義
# ─────────────────────────────
class CustomIterableDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, seq_len, transform_fn):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.transform_fn = transform_fn

    def __iter__(self):
        buffer = []
        for sample in self.dataset:
            text = self.transform_fn(sample)
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(token_ids)
            while len(buffer) >= self.seq_len:
                block = buffer[:self.seq_len]
                yield {
                    "input_ids": torch.tensor(block, dtype=torch.long),
                    "labels": torch.tensor(block, dtype=torch.long)
                }
                buffer = buffer[self.seq_len:]

# ─────────────────────────────
# wandb 初期化（local_rank == 0 の場合のみ）
# ─────────────────────────────
local_rank = int(os.environ.get("LOCAL_RANK", 0))
if local_rank == 0:
    import wandb
    WANDB_AVAILABLE = True
    PROJECT_NAME = 'titans-mac-transformer-0.3B'
    RUN_NAME = f'titans-mac-transformer-0.3B-base-Ja(03/23)-ZERO'
    WANDB_ONLINE = True
    wandb.init(project=PROJECT_NAME, mode='disabled' if not WANDB_ONLINE else 'online')
    wandb.run.name = RUN_NAME
    wandb.run.save()
else:
    WANDB_AVAILABLE = False

# ※検証用データセット（SyntheticText の先頭5件から1ブロック）
def get_validation_blocks(hf_dataset, tokenizer, seq_len, max_blocks=1):
    blocks = []
    buffer = []
    for sample in hf_dataset:
        text = sample.get("output_text", "")
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        buffer.extend(token_ids)
        while len(buffer) >= seq_len and len(blocks) < max_blocks:
            block = buffer[:seq_len]
            blocks.append({
                "input_ids": torch.tensor(block, dtype=torch.long),
                "labels": torch.tensor(block, dtype=torch.long)
            })
            buffer = buffer[seq_len:]
        if len(blocks) >= max_blocks:
            break
    return blocks
def clean_text(text):
    # 削除対象のパターンをリストにまとめる
    patterns = ["参考文献：", "**Related Topics:**", "**Referense:**"]
    # 各パターンが存在する位置を調べる（見つからない場合は -1 となるので除外）
    cut_positions = [text.find(p) for p in patterns if text.find(p) != -1]
    # もし1つ以上のパターンが見つかった場合、最も早い位置でテキストを切る
    if cut_positions:
        cut_pos = min(cut_positions)
        return text[:cut_pos].strip()
    else:
        return text


dataset_text_val = load_dataset(
    "kanhatakeyama/SyntheticText",
    split="train",
    streaming=True
)
val_samples = list(islice(dataset_text_val, 5))
print("#Loading val ")
val_blocks = get_validation_blocks(val_samples, tokenizer, SEQ_LEN, max_blocks=1)
val_loader = DataLoader(val_blocks, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)

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
    neural_mem_gate_attn_output=NEURAL_MEM_GATE_ATTENTION_OUTPUT,
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

# ⑦.5 ウォームアップ＋コサイン減衰スケジューラーの設定
WARMUP_RATIO = 0.1
warmup_steps = int(NUM_BATCHES * WARMUP_RATIO)
scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=NUM_BATCHES)

# ─────────────────────────────
# safe_decode 関数（デコード時のエラーハンドリング用）
# ─────────────────────────────
def safe_decode(token_list):
    try:
        return tokenizer.decode(token_list, skip_special_tokens=False)
    except Exception as e:
        s = "".join([chr(max(32, t)) for t in token_list])
        return s.encode("utf-8", "replace").decode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters()
    )
    # # ★ 学習済みのチェックポイントを読み込む
    # checkpoint_dir = "./KiTitans-MAC-Transformer-0.3B-base-SyntheticText"
    checkpoint_dir = "./KiTitans-MAC-Transformer-0.3B-base-SyntheticWiki"
    tag = "epoch0"
    model_engine.load_checkpoint(checkpoint_dir, tag=tag)
    print(f"Checkpoint '{tag}' loaded from {checkpoint_dir}")

    # エポック設定：epoch0 は10000ステップ、epoch1 は15000ステップ
    NUM_EPOCHS = 2

    global_step = 0
    for epoch in range(NUM_EPOCHS):
        if epoch == 0:
            # epoch0: seq_len を2048、バッチサイズを4、ステップ数を10000に設定
            new_seq_len = 1024
            bs = 5
            num_steps = 10000
            current_dataset = dataset_wiki_stream
            epoch_model_dir = "./KiTitans-MAC-Transformer-0.3B-base-SyntheticWiki"
            print("Epoch 0: Using wiki data with seq_len =", new_seq_len, ", batch_size =", bs, ", steps =", num_steps)
            print("skip the Epoch 0 (dataset = Synthetic Wiki)")
            continue
        elif epoch == 1:
            # epoch1: seq_len を4096、バッチサイズを3、ステップ数を15000に設定
            new_seq_len = 1024
            bs = 5
            num_steps = 15000
            current_dataset = dataset_cc_stream
            epoch_model_dir = "./KiTitans-MAC-Transformer-0.3B-base-SyntheticCC"
            print("Epoch 1: Using CC data with seq_len =", new_seq_len, ", batch_size =", bs, ", steps =", num_steps)

        current_iterable = CustomIterableDataset(
        current_dataset,
        tokenizer,
        new_seq_len,
        lambda sample: clean_text(sample["text"])
    )
        current_train_loader = DataLoader(current_iterable, batch_size=bs, num_workers=0, pin_memory=True)
        train_iter = iter(current_train_loader)

        for i in tqdm.tqdm(range(num_steps), mininterval=50., desc=f'training epoch {epoch}'):
            model_engine.train()

            for __ in range(GRADIENT_ACCUMULATE_EVERY):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(current_train_loader)
                    batch = next(train_iter)

                input_ids = batch["input_ids"].to(model_engine.device)
                loss = model_engine(input_ids, return_loss=True)
                model_engine.backward(loss)

            # 勾配クリッピング（最大ノルム1.0）
            torch.nn.utils.clip_grad_norm_(model_engine.module.parameters(), max_norm=1.0)

            total_norm = 0.0
            for p in model_engine.module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm += param_norm * param_norm
            grad_norm = total_norm ** 0.5

            if local_rank == 0:
                loss_val = loss.item()
                perplexity = math.exp(loss_val) if loss_val < 20 else float('inf')
                current_lr = optimizer.param_groups[0]['lr']
                if global_step % 10 == 0:
                    print(f"training loss: {loss_val}, perplexity: {perplexity}, lr: {current_lr}, grad_norm: {grad_norm}")
                if WANDB_AVAILABLE:
                    wandb.log({
                        'loss': loss_val,
                        'perplexity': perplexity,
                        'lr': current_lr,
                        'grad_norm': grad_norm,
                        'epoch': epoch,
                        'global_step': global_step
                    })
            model_engine.step()
            scheduler.step()


            if global_step % VALIDATE_EVERY == 0:
                model_engine.eval()
                with torch.no_grad():
                    val_batch = next(iter(val_loader))
                    val_input_ids = val_batch["input_ids"].to(model_engine.device)
                    val_loss = model_engine(val_input_ids, return_loss=True)
                if local_rank == 0:
                    val_loss_val = val_loss.item()
                    val_perplexity = math.exp(val_loss_val) if val_loss_val < 20 else float('inf')
                    print(f"validation loss: {val_loss_val}, val_perplexity: {val_perplexity}")
                    if WANDB_AVAILABLE:
                        wandb.log({
                            'val_loss': val_loss_val,
                            'val_perplexity': val_perplexity,
                            'epoch': epoch,
                            'global_step': global_step
                        })

            if SHOULD_GENERATE and global_step % GENERATE_EVERY == 0:
                model_engine.eval()
                prompt = "こんにちは，"
                inp_ids = tokenizer.encode(prompt, add_special_tokens=False)
                inp = torch.tensor(inp_ids, dtype=torch.long).to(model_engine.device, non_blocking=True)
                if local_rank == 0:
                    print(f"{prompt} \n\n {'*' * 100}")
                sample = model_engine.module.sample(inp[None, ...], GENERATE_LENGTH, use_cache=False)
                output_str = safe_decode(sample[0].tolist())
                if local_rank == 0:
                    print(output_str)
                    if WANDB_AVAILABLE:
                        wandb.log({
                            "gen_text": wandb.Html(output_str),
                            "epoch": epoch,
                            "global_step": global_step
                        })

            if global_step % 1000 == 0 and global_step != 0:
                ckpt_dir = "./checkpoints"
                os.makedirs(ckpt_dir, exist_ok=True)
                existing_checkpoints = [
                    d for d in os.listdir(ckpt_dir)
                    if d.startswith("step-") and os.path.isdir(os.path.join(ckpt_dir, d))
                ]
                if len(existing_checkpoints) >= 12:
                    existing_checkpoints = sorted(existing_checkpoints, key=lambda x: int(x.split('-')[-1]))
                    oldest_checkpoint = existing_checkpoints[0]
                    shutil.rmtree(os.path.join(ckpt_dir, oldest_checkpoint))
                    print(f"Removed oldest checkpoint: {oldest_checkpoint}")

                model_engine.save_checkpoint(ckpt_dir, tag=f"step-{global_step}")
                print(f"Saved checkpoint: step-{global_step}")

            global_step += 1

        os.makedirs(epoch_model_dir, exist_ok=True)
        model_engine.save_checkpoint(epoch_model_dir, tag=f"epoch{epoch}")
        print("Epoch", epoch, "model saved to", epoch_model_dir)

    final_model_dir = "./KiTitans-MAC-Transformer-0.3B-base-0320"
    os.makedirs(final_model_dir, exist_ok=True)
    model_engine.save_checkpoint(final_model_dir, tag="final")
    print("Final model saved to", final_model_dir)


if __name__ == "__main__":
    main()
