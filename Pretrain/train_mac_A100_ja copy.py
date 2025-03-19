# import argparse
# import json
# import random
# import numpy as np
# import tqdm
# import os
# import shutil
# import math
# from itertools import chain

# import torch
# from torch.utils.data import DataLoader, IterableDataset

# # huggingface datasets
# from datasets import load_dataset

# # AutoTokenizer に加え、cosineスケジューラー用の関数をインポート
# from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# from adam_atan2_pytorch import AdoptAtan2
# from titans_pytorch import (
#     MemoryAsContextTransformer,
#     MemoryMLP,
#     MemoryAttention
# )

# import deepspeed
# # ─────────────────────────────
# # データセットキャッシュ用ディレクトリ設定
# # ─────────────────────────────
# DATA_DIR = "./data/hf_datasets_pretrain"
# os.makedirs(DATA_DIR, exist_ok=True)

# # ─────────────────────────────
# # ① トークナイザーの読み込み（fast tokenizer を使用）
# # ─────────────────────────────
# tokenizer = AutoTokenizer.from_pretrained("weblab-GENIAC/Tanuki-8B-dpo-v1.0", use_fast=True)
# NUM_TOKENS = tokenizer.vocab_size

# # ─────────────────────────────
# # ② ハイパーパラメータ設定
# # ─────────────────────────────
# # 推定処理をコメントアウトし、NUM_BATCHES をハードコード
# # NUM_BATCHES = 55000  
# BATCH_SIZE = 4
# GRADIENT_ACCUMULATE_EVERY = 5
# LEARNING_RATE = 2e-4
# VALIDATE_EVERY  = 200
# GENERATE_EVERY  = 500
# PRIME_LENGTH = 100
# GENERATE_LENGTH = 512
# SHOULD_GENERATE = True
# SEQ_LEN = 1024

# # neural memory 関連
# NEURAL_MEMORY_DEPTH = 2
# NUM_PERSIST_MEM = 16
# NUM_LONGTERM_MEM = 16
# NEURAL_MEM_LAYERS = tuple(range(2, 15, 2))
# NEURAL_MEM_GATE_ATTN_OUTPUT = False
# NEURAL_MEM_MOMENTUM = True
# NEURAL_MEM_MOMENTUM_ORDER = 1
# NEURAL_MEM_QK_NORM = True
# NEURAL_MEM_MAX_LR = 1e-3
# USE_MEM_ATTENTION_MODEL = False
# WINDOW_SIZE = 128
# NEURAL_MEM_SEGMENT_LEN = 16
# NEURAL_MEM_BATCH_SIZE = 128
# SLIDING_WINDOWS = True
# STORE_ATTN_POOL_CHUNKS = True
# MEMORY_MODEL_PER_LAYER_LEARNED_LR = True
# NEURAL_MEM_WEIGHT_RESIDUAL = True
# NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True

# MODEL_DIM = 1024
# MODEL_DEPTH = 15

# # ─────────────────────────────
# # ②.5 推定処理部分（コメントアウト）
# # ※以下の推定処理を使用せず、NUM_BATCHES をハードコードします。
# #
# def flat_map_iterable(dataset, fn):
#     for sample in dataset:
#         for item in fn(sample):
#             yield item

# def transform_synthetic_text(sample):
#     return sample.get("output_text", "")

# def transform_wiki_est(sample):
#     return [sample.get("ja", ""), sample.get("eng", "")]

# def transform_synthetic_text_cc(sample):
#     return sample.get("ja", "")

# def estimate_total_blocks_custom_streaming(dataset_list, tokenizer, seq_len, sample_size=100):
#     total_estimated_tokens = 0
#     for dataset, transform, default_total in dataset_list:
#         try:
#             num_samples = dataset.info.splits["train"].num_examples
#         except Exception as e:
#             if default_total is not None:
#                 num_samples = default_total
#             else:
#                 raise ValueError("streaming dataset の総サンプル数が取得できませんでした。info 属性を確認してください。")
        
#         samples = []
#         for i, sample in enumerate(dataset):
#             samples.append(sample)
#             if i + 1 >= sample_size:
#                 break

#         total_tokens = 0
#         total_count = 0
#         for sample in samples:
#             result = transform(sample)
#             if isinstance(result, list):
#                 for x in result:
#                     total_tokens += len(tokenizer.encode(x, add_special_tokens=False))
#                     total_count += 1
#             else:
#                 total_tokens += len(tokenizer.encode(result, add_special_tokens=False))
#                 total_count += 1
#         avg_tokens = total_tokens / total_count if total_count > 0 else 0

#         count_list = []
#         for sample in samples:
#             result = transform(sample)
#             if isinstance(result, list):
#                 count_list.append(len(result))
#             else:
#                 count_list.append(1)
#         avg_count = sum(count_list) / len(count_list) if count_list else 1

#         effective_total_samples = num_samples * avg_count
#         estimated_tokens = avg_tokens * effective_total_samples
#         total_estimated_tokens += estimated_tokens

#     total_blocks = total_estimated_tokens // seq_len
#     return int(total_blocks)

# # 推定用データセットのロード処理（streaming）
# if local_rank == 0:
#     dataset_text_est = load_dataset("kanhatakeyama/SyntheticText", split="train", streaming=True, cache_dir=DATA_DIR, download_mode="reuse_cache_if_exists")
#     # dataset_wiki_est = load_dataset("kanhatakeyama/SyntheticTextWikiTranslate", split="train", streaming=True, cache_dir=DATA_DIR, download_mode="reuse_cache_if_exists")
#     # dataset_cc_est = load_dataset("kanhatakeyama/SyntheticTextCC", split="train", streaming=True, cache_dir=DATA_DIR, download_mode="reuse_cache_if_exists")
#     datasets_for_estimation = [
#         (dataset_text_est, transform_synthetic_text, 3400000),
#         # (dataset_wiki_est, transform_wiki_est, 2100000),
#         # (dataset_cc_est, transform_synthetic_text_cc, 2380000)
#     ]
#     total_blocks = estimate_total_blocks_custom_streaming(datasets_for_estimation, tokenizer, SEQ_LEN, sample_size=100)
#     with open("total_blocks.txt", "w") as f:
#         f.write(str(total_blocks))
# else:
#     import time
#     while not os.path.exists("total_blocks.txt"):
#         time.sleep(1)
#     with open("total_blocks.txt", "r") as f:
#         total_blocks = int(f.read())

# NUM_BATCHES = total_blocks // (GRADIENT_ACCUMULATE_EVERY * BATCH_SIZE)

# #NUM_BATCHES = 2642033
# print(f"NUM_BATCHES (1エポックあたりの最適化ステップ数){NUM_BATCHES} ")

# # ─────────────────────────────
# # ③ 各データセット（ストリーミング版）のロードと変換処理
# # ─────────────────────────────

# # SyntheticText： "output_text" を使用しシャッフル
# dataset_text_stream = load_dataset(
#     "kanhatakeyama/SyntheticText",
#     split="train",
#     streaming=True,
#     cache_dir=DATA_DIR,
#     download_mode="reuse_cache_if_exists"
# ).shuffle(buffer_size=1000)

# # SyntheticTextWikiTranslate： flat_map_iterable を用いて、各サンプルから "ja" と "eng" を個別のサンプルとして展開しシャッフル
# def flat_map_iterable(dataset, fn):
#     for sample in dataset:
#         for item in fn(sample):
#             yield item

# raw_wiki_stream = load_dataset(
#     "kanhatakeyama/SyntheticTextWikiTranslate",
#     split="train",
#     streaming=True,
#     cache_dir=DATA_DIR,
#     download_mode="reuse_cache_if_exists"
# ).shuffle(buffer_size=1000)
# dataset_wiki_stream = flat_map_iterable(raw_wiki_stream, lambda sample: [{"text": sample.get("ja", "")}, {"text": sample.get("eng", "")}])

# # SyntheticTextCC： "ja" 列を使用（シャッフル）
# dataset_cc_stream = load_dataset(
#     "kanhatakeyama/SyntheticTextCC",
#     split="train",
#     streaming=True,
#     cache_dir=DATA_DIR,
#     download_mode="reuse_cache_if_exists"
# ).shuffle(buffer_size=1000)

# # ─────────────────────────────
# # ③-2 カスタム IterableDataset の定義
# # ─────────────────────────────
# class CustomIterableDataset(IterableDataset):
#     def __init__(self, dataset, tokenizer, seq_len, transform_fn):
#         super().__init__()
#         self.dataset = dataset
#         self.tokenizer = tokenizer
#         self.seq_len = seq_len
#         self.transform_fn = transform_fn

#     def __iter__(self):
#         buffer = []
#         for sample in self.dataset:
#             text = self.transform_fn(sample)
#             token_ids = self.tokenizer.encode(text, add_special_tokens=False)
#             buffer.extend(token_ids)
#             while len(buffer) >= self.seq_len:
#                 block = buffer[:self.seq_len]
#                 yield {
#                     "input_ids": torch.tensor(block, dtype=torch.long),
#                     "labels": torch.tensor(block, dtype=torch.long)
#                 }
#                 buffer = buffer[self.seq_len:]
# # wandb 初期化（local_rank == 0 の場合のみ）
# local_rank = int(os.environ.get("LOCAL_RANK", 0))
# if local_rank == 0:
#     import wandb
#     WANDB_AVAILABLE = True
#     PROJECT_NAME = 'titans-mac-transformer-0.3B'
#     RUN_NAME = f'titans-mac-transformer-0.3B-base-ja-curricuram {16} longterm mems, layers {(2,4,6,8,10,12,14)}'
#     WANDB_ONLINE = True
#     wandb.init(project=PROJECT_NAME, mode='disabled' if not WANDB_ONLINE else 'online')
#     wandb.run.name = RUN_NAME
#     wandb.run.save()
# else:
#     WANDB_AVAILABLE = False
# # 各データセットごとに IterableDataset を作成
# iterable_text = CustomIterableDataset(dataset_text_stream, tokenizer, SEQ_LEN, lambda sample: sample.get("output_text", ""))
# # iterable_wiki = CustomIterableDataset(dataset_wiki_stream, tokenizer, SEQ_LEN, lambda sample: sample["text"])
# # iterable_cc = CustomIterableDataset(dataset_cc_stream, tokenizer, SEQ_LEN, lambda sample: sample.get("ja", ""))

# # combined_iterable = chain(iterable_text, iterable_wiki, iterable_cc)

# train_loader = DataLoader(iterable_text, batch_size=BATCH_SIZE)

# # ─────────────────────────────
# # ④ 検証用データセット（SyntheticText の先頭500件から1ブロック）
# # ─────────────────────────────
# def get_validation_blocks(hf_dataset, tokenizer, seq_len, max_blocks=10):
#     blocks = []
#     buffer = []
#     for sample in hf_dataset:
#         text = sample.get("output_text", "")
#         token_ids = tokenizer.encode(text, add_special_tokens=False)
#         buffer.extend(token_ids)
#         while len(buffer) >= seq_len and len(blocks) < max_blocks:
#             block = buffer[:seq_len]
#             blocks.append({
#                 "input_ids": torch.tensor(block, dtype=torch.long),
#                 "labels": torch.tensor(block, dtype=torch.long)
#             })
#             buffer = buffer[seq_len:]
#         if len(blocks) >= max_blocks:
#             break
#     return blocks

# dataset_text_val = load_dataset(
#     "kanhatakeyama/SyntheticText",
#     split="train",
#     streaming=False,
#     cache_dir=DATA_DIR,
#     download_mode="reuse_cache_if_exists"
# ).select(range(500))
# val_blocks = get_validation_blocks(dataset_text_val, tokenizer, SEQ_LEN, max_blocks=1)
# val_loader = DataLoader(val_blocks, batch_size=BATCH_SIZE)

# # ─────────────────────────────
# # ⑤ memory model の設定
# # ─────────────────────────────
# if USE_MEM_ATTENTION_MODEL:
#     neural_memory_model = MemoryAttention(dim=64)
# else:
#     neural_memory_model = MemoryMLP(dim=64, depth=NEURAL_MEMORY_DEPTH)

# # ─────────────────────────────
# # ⑥ MemoryAsContextTransformer のインスタンス作成
# # ─────────────────────────────
# model = MemoryAsContextTransformer(
#     num_tokens=NUM_TOKENS,
#     dim=MODEL_DIM,
#     depth=MODEL_DEPTH,
#     segment_len=WINDOW_SIZE,
#     num_persist_mem_tokens=NUM_PERSIST_MEM,
#     num_longterm_mem_tokens=NUM_LONGTERM_MEM,
#     neural_memory_layers=NEURAL_MEM_LAYERS,
#     neural_memory_segment_len=NEURAL_MEM_SEGMENT_LEN,
#     neural_memory_batch_size=NEURAL_MEM_BATCH_SIZE,
#     neural_mem_gate_attn_output=NEURAL_MEM_GATE_ATTN_OUTPUT,
#     neural_mem_weight_residual=NEURAL_MEM_WEIGHT_RESIDUAL,
#     neural_memory_qkv_receives_diff_views=NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW,
#     use_flex_attn=True,
#     sliding_window_attn=SLIDING_WINDOWS,
#     neural_memory_model=neural_memory_model,
#     neural_memory_kwargs=dict(
#         dim_head=64,
#         heads=16,
#         attn_pool_chunks=STORE_ATTN_POOL_CHUNKS,
#         qk_rmsnorm=NEURAL_MEM_QK_NORM,
#         momentum=NEURAL_MEM_MOMENTUM,
#         momentum_order=NEURAL_MEM_MOMENTUM_ORDER,
#         default_step_transform_max_lr=NEURAL_MEM_MAX_LR,
#         use_accelerated_scan=True,
#         per_parameter_lr_modulation=MEMORY_MODEL_PER_LAYER_LEARNED_LR
#     )
# ).cuda()

# # ─────────────────────────────
# # ⑦ オプティマイザの設定
# # ─────────────────────────────
# optim = AdoptAtan2(model.parameters(), lr=LEARNING_RATE)

# # ⑦.5 ウォームアップ＋コサイン減衰スケジューラーの設定
# WARMUP_RATIO = 0.1
# warmup_steps = int(NUM_BATCHES * WARMUP_RATIO)
# scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=NUM_BATCHES)

# # ─────────────────────────────
# # safe_decode 関数（デコード時のエラーハンドリング用）
# # ─────────────────────────────
# def safe_decode(token_list):
#     try:
#         return tokenizer.decode(token_list, skip_special_tokens=False)
#     except Exception as e:
#         s = "".join([chr(max(32, t)) for t in token_list])
#         return s.encode("utf-8", "replace").decode("utf-8")

# # ─────────────────────────────
# # ⑧ DeepSpeed による初期化と学習ループ
# # ─────────────────────────────
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--local_rank", type=int, default=0)
#     parser = deepspeed.add_config_arguments(parser)
#     args = parser.parse_args()

#     model_engine, optimizer, _, _ = deepspeed.initialize(
#         args=args,
#         model=model,
#         model_parameters=model.parameters()
#     )

#     train_iter = iter(train_loader)

#     for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=50., desc='training'):
#         model_engine.train()

#         for __ in range(GRADIENT_ACCUMULATE_EVERY):
#             try:
#                 batch = next(train_iter)
#             except StopIteration:
#                 train_iter = iter(train_loader)
#                 batch = next(train_iter)

#             input_ids = batch["input_ids"].to(model_engine.device)
#             loss = model_engine(input_ids, return_loss=True)
#             model_engine.backward(loss)

#         model_engine.step()
#         scheduler.step()

#         if local_rank == 0:
#             loss_val = loss.item()
#             perplexity = math.exp(loss_val) if loss_val < 20 else float('inf')
#             current_lr = optimizer.param_groups[0]['lr']

#             total_norm = 0.0
#             for p in model_engine.module.parameters():
#                 if p.grad is not None:
#                     param_norm = p.grad.data.norm(2).item()
#                     total_norm += param_norm * param_norm
#             grad_norm = total_norm ** 0.5

#             print(f"training loss: {loss_val}, perplexity: {perplexity}, lr: {current_lr}, grad_norm: {grad_norm}")
#             if WANDB_AVAILABLE:
#                 wandb.log({
#                     'loss': loss_val,
#                     'perplexity': perplexity,
#                     'lr': current_lr,
#                     'grad_norm': grad_norm
#                 })

#         if i % VALIDATE_EVERY == 0:
#             model_engine.eval()
#             with torch.no_grad():
#                 val_batch = next(iter(val_loader))
#                 val_input_ids = val_batch["input_ids"].to(model_engine.device)
#                 val_loss = model_engine(val_input_ids, return_loss=True)
#             if local_rank == 0:
#                 val_loss_val = val_loss.item()
#                 val_perplexity = math.exp(val_loss_val) if val_loss_val < 20 else float('inf')
#                 print(f"validation loss: {val_loss_val}, val_perplexity: {val_perplexity}")
#                 if WANDB_AVAILABLE:
#                     wandb.log({
#                         'val_loss': val_loss_val,
#                         'val_perplexity': val_perplexity
#                     })

#         if SHOULD_GENERATE and i % GENERATE_EVERY == 0:
#             model_engine.eval()
#             inp = random.choice(val_blocks)["input_ids"][:PRIME_LENGTH].to(model_engine.device, non_blocking=True)
#             prime = safe_decode(inp.tolist())
#             if local_rank == 0:
#                 print(f"{prime} \n\n {'*' * 100}")
#             sample = model_engine.module.sample(inp[None, ...], GENERATE_LENGTH, use_cache=False)
#             output_str = safe_decode(sample[0].tolist())
#             if local_rank == 0:
#                 print(output_str)
#                 if WANDB_AVAILABLE:
#                     wandb.log({"generated_text": output_str})

#         if i % 1000 == 0 and i != 0:
#             checkpoint_dir = "./checkpoints"
#             os.makedirs(checkpoint_dir, exist_ok=True)
#             existing_checkpoints = [
#                 d for d in os.listdir(checkpoint_dir)
#                 if d.startswith("step-") and os.path.isdir(os.path.join(checkpoint_dir, d))
#             ]
#             if len(existing_checkpoints) >= 12:
#                 existing_checkpoints = sorted(existing_checkpoints, key=lambda x: int(x.split('-')[-1]))
#                 oldest_checkpoint = existing_checkpoints[0]
#                 shutil.rmtree(os.path.join(checkpoint_dir, oldest_checkpoint))
#                 print(f"Removed oldest checkpoint: {oldest_checkpoint}")

#             model_engine.save_checkpoint(checkpoint_dir, tag=f"step-{i}")
#             print(f"Saved checkpoint: step-{i}")

# if __name__ == "__main__":
#     main()
import argparse
import json
import random
import numpy as np
import tqdm
import os
import shutil
import math
from itertools import chain

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
# データセットキャッシュ用ディレクトリ設定
# ─────────────────────────────
DATA_DIR = "./data/hf_datasets_pretrain"
os.makedirs(DATA_DIR, exist_ok=True)

# ─────────────────────────────
# ① トークナイザーの読み込み（fast tokenizer を使用）
# ─────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("weblab-GENIAC/Tanuki-8B-dpo-v1.0", use_fast=True)
NUM_TOKENS = tokenizer.vocab_size

# ─────────────────────────────
# ② ハイパーパラメータ設定
# ─────────────────────────────
# 推定処理をコメントアウトし、NUM_BATCHES をハードコード
# NUM_BATCHES = 55000  
BATCH_SIZE = 4
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
# ②.5 推定処理部分（コメントアウト）
# ※以下の推定処理を使用せず、NUM_BATCHES をハードコードします。
#
# def flat_map_iterable(dataset, fn):
#     for sample in dataset:
#         for item in fn(sample):
#             yield item

# def transform_synthetic_text(sample):
#     return sample.get("output_text", "")

# def transform_wiki_est(sample):
#     return [sample.get("ja", ""), sample.get("eng", "")]

# def transform_synthetic_text_cc(sample):
#     return sample.get("ja", "")

# def estimate_total_blocks_custom_streaming(dataset_list, tokenizer, seq_len, sample_size=100):
#     total_estimated_tokens = 0
#     for dataset, transform, default_total in dataset_list:
#         try:
#             num_samples = dataset.info.splits["train"].num_examples
#         except Exception as e:
#             if default_total is not None:
#                 num_samples = default_total
#             else:
#                 raise ValueError("streaming dataset の総サンプル数が取得できませんでした。info 属性を確認してください。")
        
#         samples = []
#         for i, sample in enumerate(dataset):
#             samples.append(sample)
#             if i + 1 >= sample_size:
#                 break

#         total_tokens = 0
#         total_count = 0
#         for sample in samples:
#             result = transform(sample)
#             if isinstance(result, list):
#                 for x in result:
#                     total_tokens += len(tokenizer.encode(x, add_special_tokens=False))
#                     total_count += 1
#             else:
#                 total_tokens += len(tokenizer.encode(result, add_special_tokens=False))
#                 total_count += 1
#         avg_tokens = total_tokens / total_count if total_count > 0 else 0

#         count_list = []
#         for sample in samples:
#             result = transform(sample)
#             if isinstance(result, list):
#                 count_list.append(len(result))
#             else:
#                 count_list.append(1)
#         avg_count = sum(count_list) / len(count_list) if count_list else 1

#         effective_total_samples = num_samples * avg_count
#         estimated_tokens = avg_tokens * effective_total_samples
#         total_estimated_tokens += estimated_tokens

#     total_blocks = total_estimated_tokens // seq_len
#     return int(total_blocks)

# # 推定用データセットのロード処理（streaming）
# if local_rank == 0:
#     dataset_text_est = load_dataset("kanhatakeyama/SyntheticText", split="train", streaming=True, cache_dir=DATA_DIR, download_mode="reuse_cache_if_exists")
#     # dataset_wiki_est = load_dataset("kanhatakeyama/SyntheticTextWikiTranslate", split="train", streaming=True, cache_dir=DATA_DIR, download_mode="reuse_cache_if_exists")
#     # dataset_cc_est = load_dataset("kanhatakeyama/SyntheticTextCC", split="train", streaming=True, cache_dir=DATA_DIR, download_mode="reuse_cache_if_exists")
#     datasets_for_estimation = [
#         (dataset_text_est, transform_synthetic_text, 3400000),
#         # (dataset_wiki_est, transform_wiki_est, 2100000),
#         # (dataset_cc_est, transform_synthetic_text_cc, 2380000)
#     ]
#     total_blocks = estimate_total_blocks_custom_streaming(datasets_for_estimation, tokenizer, SEQ_LEN, sample_size=100)
#     with open("total_blocks.txt", "w") as f:
#         f.write(str(total_blocks))
# else:
#     import time
#     while not os.path.exists("total_blocks.txt"):
#         time.sleep(1)
#     with open("total_blocks.txt", "r") as f:
#         total_blocks = int(f.read())

# NUM_BATCHES = total_blocks // (GRADIENT_ACCUMULATE_EVERY * BATCH_SIZE)

#NUM_BATCHES = 2642033
# print(f"NUM_BATCHES (1エポックあたりの最適化ステップ数){NUM_BATCHES} ")
NUM_BATCHES = 50000
# ─────────────────────────────
# ③ 各データセット（ストリーミング版）のロードと変換処理
# ─────────────────────────────

# SyntheticText： "output_text" を使用しシャッフル
dataset_text_stream = load_dataset(
    "kanhatakeyama/SyntheticText",
    split="train",
    streaming=True,
    cache_dir=DATA_DIR,
    download_mode="reuse_cache_if_exists"
)

# # SyntheticTextWikiTranslate： flat_map_iterable を用いて、各サンプルから "ja" と "eng" を個別のサンプルとして展開しシャッフル
# def flat_map_iterable(dataset, fn):
#     for sample in dataset:
#         for item in fn(sample):
#             yield item

# raw_wiki_stream = load_dataset(
#     "kanhatakeyama/SyntheticTextWikiTranslate",
#     split="train",
#     streaming=True,
#     cache_dir=DATA_DIR,
#     download_mode="reuse_cache_if_exists"
# ).shuffle(buffer_size=1000)
# dataset_wiki_stream = flat_map_iterable(raw_wiki_stream, lambda sample: [{"text": sample.get("ja", "")}, {"text": sample.get("eng", "")}])

# # SyntheticTextCC： "ja" 列を使用（シャッフル）
# dataset_cc_stream = load_dataset(
#     "kanhatakeyama/SyntheticTextCC",
#     split="train",
#     streaming=True,
#     cache_dir=DATA_DIR,
#     download_mode="reuse_cache_if_exists"
# ).shuffle(buffer_size=1000)

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
# wandb 初期化（local_rank == 0 の場合のみ）
local_rank = int(os.environ.get("LOCAL_RANK", 0))
if local_rank == 0:
    import wandb
    WANDB_AVAILABLE = True
    PROJECT_NAME = 'titans-mac-transformer-0.3B'
    RUN_NAME = f'titans-mac-transformer-0.3B-base-ja-curricuram {16} longterm mems, layers {(2,4,6,8,10,12,14)}'
    WANDB_ONLINE = True
    wandb.init(project=PROJECT_NAME, mode='disabled' if not WANDB_ONLINE else 'online')
    wandb.run.name = RUN_NAME
    wandb.run.save()
else:
    WANDB_AVAILABLE = False
# 各データセットごとに IterableDataset を作成
iterable_text = CustomIterableDataset(dataset_text_stream, tokenizer, SEQ_LEN, lambda sample: sample.get("output_text", ""))
# iterable_wiki = CustomIterableDataset(dataset_wiki_stream, tokenizer, SEQ_LEN, lambda sample: sample["text"])
# iterable_cc = CustomIterableDataset(dataset_cc_stream, tokenizer, SEQ_LEN, lambda sample: sample.get("ja", ""))

# combined_iterable = chain(iterable_text, iterable_wiki, iterable_cc)

train_loader = DataLoader(iterable_text, batch_size=BATCH_SIZE)

# ─────────────────────────────
# ④ 検証用データセット（SyntheticText の先頭500件から1ブロック）
# ─────────────────────────────
def get_validation_blocks(hf_dataset, tokenizer, seq_len, max_blocks=10):
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

dataset_text_val = load_dataset(
    "kanhatakeyama/SyntheticText",
    split="train",
    streaming=False,
    cache_dir=DATA_DIR,
    download_mode="reuse_cache_if_exists"
).select(range(500))
val_blocks = get_validation_blocks(dataset_text_val, tokenizer, SEQ_LEN, max_blocks=1)
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

# ─────────────────────────────
# ⑧ DeepSpeed による初期化と学習ループ
# ─────────────────────────────
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

    train_iter = iter(train_loader)

    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=50., desc='training'):
        model_engine.train()

        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(model_engine.device)
            loss = model_engine(input_ids, return_loss=True)
            model_engine.backward(loss)

        model_engine.step()
        scheduler.step()

        if local_rank == 0:
            loss_val = loss.item()
            perplexity = math.exp(loss_val) if loss_val < 20 else float('inf')
            current_lr = optimizer.param_groups[0]['lr']

            total_norm = 0.0
            for p in model_engine.module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm += param_norm * param_norm
            grad_norm = total_norm ** 0.5

            print(f"training loss: {loss_val}, perplexity: {perplexity}, lr: {current_lr}, grad_norm: {grad_norm}")
            if WANDB_AVAILABLE:
                wandb.log({
                    'loss': loss_val,
                    'perplexity': perplexity,
                    'lr': current_lr,
                    'grad_norm': grad_norm
                })

        if i % VALIDATE_EVERY == 0:
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
                        'val_perplexity': val_perplexity
                    })

        if SHOULD_GENERATE and i % GENERATE_EVERY == 0:
            model_engine.eval()
            inp = random.choice(val_blocks)["input_ids"][:PRIME_LENGTH].to(model_engine.device, non_blocking=True)
            prime = safe_decode(inp.tolist())
            if local_rank == 0:
                print(f"{prime} \n\n {'*' * 100}")
            sample = model_engine.module.sample(inp[None, ...], GENERATE_LENGTH, use_cache=False)
            output_str = safe_decode(sample[0].tolist())
            if local_rank == 0:
                print(output_str)
                if WANDB_AVAILABLE:
                    wandb.log({"generated_text": output_str})

        if i % 1000 == 0 and i != 0:
            checkpoint_dir = "./checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            existing_checkpoints = [
                d for d in os.listdir(checkpoint_dir)
                if d.startswith("step-") and os.path.isdir(os.path.join(checkpoint_dir, d))
            ]
            if len(existing_checkpoints) >= 12:
                existing_checkpoints = sorted(existing_checkpoints, key=lambda x: int(x.split('-')[-1]))
                oldest_checkpoint = existing_checkpoints[0]
                shutil.rmtree(os.path.join(checkpoint_dir, oldest_checkpoint))
                print(f"Removed oldest checkpoint: {oldest_checkpoint}")

            model_engine.save_checkpoint(checkpoint_dir, tag=f"step-{i}")
            print(f"Saved checkpoint: step-{i}")

if __name__ == "__main__":
    main()

