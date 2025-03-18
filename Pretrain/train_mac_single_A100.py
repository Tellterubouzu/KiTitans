import random
import tqdm
import gzip
import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer  # トークナイザー用インポート

from adam_atan2_pytorch import AdoptAtan2

from titans_pytorch import (
    MemoryAsContextTransformer,
    MemoryMLP,
    MemoryAttention
)

# ─────────────────────────────
# ① トークナイザーの読み込み
# ─────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("weblab-GENIAC/Tanuki-8B-dpo-v1.0")
NUM_TOKENS = tokenizer.vocab_size

# ─────────────────────────────
# ② ハイパーパラメータ設定（総パラメータ数約500Mを目指す）
# ─────────────────────────────
NUM_BATCHES = int(1e5)
BATCH_SIZE = 2
GRADIENT_ACCUMULATE_EVERY = 8
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
PRIME_LENGTH = 100
GENERATE_LENGTH =512
SHOULD_GENERATE = True
SEQ_LEN = 1024

# neural memory 関連の設定（パラメータ数拡大のため調整）
NEURAL_MEMORY_DEPTH = 2
NUM_PERSIST_MEM = 16
NUM_LONGTERM_MEM = 16
# 2から24まで偶数層に対して neural memory を適用
NEURAL_MEM_LAYERS = tuple(range(2, 25, 2))
NEURAL_MEM_GATE_ATTN_OUTPUT = False
NEURAL_MEM_MOMENTUM = True
NEURAL_MEM_MOMENTUM_ORDER = 1
NEURAL_MEM_QK_NORM = True
NEURAL_MEM_MAX_LR = 1e-3
USE_MEM_ATTENTION_MODEL = False
WINDOW_SIZE = 128  # セグメント長を128に増加
NEURAL_MEM_SEGMENT_LEN = 16
NEURAL_MEM_BATCH_SIZE = 128
SLIDING_WINDOWS = True
STORE_ATTN_POOL_CHUNKS = True
MEMORY_MODEL_PER_LAYER_LEARNED_LR = True
NEURAL_MEM_WEIGHT_RESIDUAL = True
NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True

# モデルの大きさ設定（0.5Bパラメータに向けて拡大）
MODEL_DIM = 1024
MODEL_DEPTH = 24

PROJECT_NAME = 'titans-mac-transformer-0.5B'
RUN_NAME = f'mac - {NUM_LONGTERM_MEM} longterm mems, layers {NEURAL_MEM_LAYERS}'
WANDB_ONLINE = True

import wandb
wandb.init(project = PROJECT_NAME, mode = 'disabled' if not WANDB_ONLINE else 'online')
wandb.run.name = RUN_NAME
wandb.run.save()

# ─────────────────────────────
# ③ ヘルパー関数
# ─────────────────────────────
def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# ─────────────────────────────
# ④ memory model の設定
# ─────────────────────────────
if USE_MEM_ATTENTION_MODEL:
    neural_memory_model = MemoryAttention(dim = 64)
else:
    neural_memory_model = MemoryMLP(dim = 64, depth = NEURAL_MEMORY_DEPTH)

# ─────────────────────────────
# ⑤ MemoryAsContextTransformer のインスタンス作成
# ─────────────────────────────
model = MemoryAsContextTransformer(
    num_tokens = NUM_TOKENS,
    dim = MODEL_DIM,
    depth = MODEL_DEPTH,
    segment_len = WINDOW_SIZE,
    num_persist_mem_tokens = NUM_PERSIST_MEM,
    num_longterm_mem_tokens = NUM_LONGTERM_MEM,
    neural_memory_layers = NEURAL_MEM_LAYERS,
    neural_memory_segment_len = NEURAL_MEM_SEGMENT_LEN,
    neural_memory_batch_size = NEURAL_MEM_BATCH_SIZE,
    neural_mem_gate_attn_output = NEURAL_MEM_GATE_ATTN_OUTPUT,
    neural_mem_weight_residual = NEURAL_MEM_WEIGHT_RESIDUAL,
    neural_memory_qkv_receives_diff_views = NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW,
    use_flex_attn = True,
    sliding_window_attn = SLIDING_WINDOWS,
    neural_memory_model = neural_memory_model,
    neural_memory_kwargs = dict(
        dim_head = 64,
        heads = 16,
        attn_pool_chunks = STORE_ATTN_POOL_CHUNKS,
        qk_rmsnorm = NEURAL_MEM_QK_NORM,
        momentum = NEURAL_MEM_MOMENTUM,
        momentum_order = NEURAL_MEM_MOMENTUM_ORDER,
        default_step_transform_max_lr = NEURAL_MEM_MAX_LR,
        use_accelerated_scan = True,
        per_parameter_lr_modulation = MEMORY_MODEL_PER_LAYER_LEARNED_LR
    )
).cuda()

# ─────────────────────────────
# ⑥ データの準備（enwik8データセット）
# ─────────────────────────────
with gzip.open('KiTitans/Pretrain/epoch_datasets/epoch0_dataset.jsonl.gz') as file:
    data = np.frombuffer(file.read(int(95e6)), dtype = np.uint8).copy()
    data_train, data_val = np.split(data, [int(90e6)])
    data_train, data_val = map(torch.from_numpy, (data_train, data_val))

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

optim = AdoptAtan2(model.parameters(), lr = LEARNING_RATE)

# ─────────────────────────────
# ⑦ fp16学習のための GradScaler の導入
# ─────────────────────────────
scaler = torch.cuda.amp.GradScaler()

# ─────────────────────────────
# ⑧ 学習ループ（自動混合精度 fp16 でトレーニング）
# ─────────────────────────────
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10., desc = 'training'):
    model.train()
    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        with torch.cuda.amp.autocast(dtype = torch.bfloat16):
            loss = model(next(train_loader), return_loss = True)
        scaler.scale(loss).backward()
    if i % VALIDATE_EVERY == 0:
        print(f'training loss: {loss.item()}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    scaler.step(optim)
    scaler.update()
    optim.zero_grad()
    wandb.log(dict(loss = loss.item()))
    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            loss = model(next(val_loader), return_loss = True)
            print(f'validation loss: {loss.item()}')
    if SHOULD_GENERATE and i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        print(f'{prime} \n\n {"*" * 100}')
        sample = model.sample(inp[None, ...], GENERATE_LENGTH, use_cache = False)
        output_str = decode_tokens(sample[0])
        output_str = output_str.encode("utf-8", "replace").decode("utf-8")
        print(output_str)
