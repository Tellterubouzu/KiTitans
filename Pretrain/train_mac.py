import random
import tqdm
import gzip
import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from adam_atan2_pytorch import AdoptAtan2

from titans_pytorch import (
    MemoryAsContextTransformer,
    MemoryMLP,
    MemoryAttention
)

# constants
NUM_BATCHES = int(1e5)              # 総トレーニングバッチ数（例: 100,000 バッチ）
BATCH_SIZE = 4                      # 1バッチあたりのサンプル数
GRADIENT_ACCUMULATE_EVERY = 4       # 勾配を更新する前に累積するバッチ数（複数バッチ分の勾配を合計して更新）
LEARNING_RATE = 2e-4                # 学習率
VALIDATE_EVERY  = 100               # 検証を行う間隔（バッチ数単位）
GENERATE_EVERY  = 500               # 生成（テキスト生成等）を行う間隔（バッチ数単位）
PRIME_LENGTH = 100                  # テキスト生成時のプライム（初期入力）シーケンスの長さ
GENERATE_LENGTH = 512               # テキスト生成時に生成するトークン数
SHOULD_GENERATE = True              # テキスト生成を実施するかどうかのフラグ
SEQ_LEN = 512                       # 入力シーケンスの長さ

# neural memory related
NEURAL_MEMORY_DEPTH = 2             # Neural Memoryモジュール内のMLPの深さ（層数）
NUM_PERSIST_MEM = 4                 # 永続メモリトークンの数（タスク固有の固定メモリ数）
NUM_LONGTERM_MEM = 4                # 長期メモリトークンの数（長期記憶として保持するトークン数）
NEURAL_MEM_LAYERS = (2, 4, 6)         # Neural Memoryを持つTransformer層の番号（例: 2層目、4層目、6層目に搭載）
NEURAL_MEM_GATE_ATTN_OUTPUT = False # アテンション出力に対してゲート処理を適用するかのフラグ
NEURAL_MEM_MOMENTUM = True          # Neural Memory更新時にモメンタム（勾配の慣性項）を使用するかのフラグ
NEURAL_MEM_MOMENTUM_ORDER = 1       # モメンタムの次数（1なら一次モメンタム）
NEURAL_MEM_QK_NORM = True           # Neural Memoryでクエリ・キーに対してRMSNormなどの正規化を適用するかのフラグ
NEURAL_MEM_MAX_LR = 1e-1            # Neural Memory更新時に使用する最大学習率（適応学習率変換用）
USE_MEM_ATTENTION_MODEL = False     # Neural MemoryモジュールとしてMemoryAttentionを使用するか、MemoryMLPを使用するかの選択フラグ
WINDOW_SIZE = 32                    # アテンションのスライディングウィンドウサイズ（セグメント長として利用）
NEURAL_MEM_SEGMENT_LEN = 4          # Neural Memoryのセグメント長（学習率やモメンタムの分解の粒度を決定）
NEURAL_MEM_BATCH_SIZE = 128         # Neural Memory更新に使用するバッチサイズ（小さい値だと頻繁に更新可能）
SLIDING_WINDOWS = True              # スライディングウィンドウアテンションを使用するかのフラグ
STORE_ATTN_POOL_CHUNKS = True       # チャンクごとのアテンションプーリングを使用するか（モメンタムやレイヤー毎の学習率調整などに影響）
MEMORY_MODEL_PER_LAYER_LEARNED_LR = True  # Neural Memory各層で個別に学習率を調整するかのフラグ
NEURAL_MEM_WEIGHT_RESIDUAL = True   # 前層のNeural Memory重みを残差接続として取り入れるか（性能向上のための追加工夫）
NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True  # Neural Memoryのクエリ・キー・バリューが異なる層の出力から取得されるようにする（NAS的アプローチ）

# experiment related
PROJECT_NAME = 'titans-mac-transformer'  # プロジェクト名（実験管理やWandBログ用）
RUN_NAME = f'mac - {NUM_LONGTERM_MEM} longterm mems, layers {NEURAL_MEM_LAYERS}'  # 実験ランの名前（使用する長期メモリトークン数や搭載層を表示）
WANDB_ONLINE = False              # WandBでオンラインログを行うかどうかのフラグ（Falseならローカルのみ）

# perf related
USE_ACCELERATED_SCAN = True       # 高速なAssociative Scanアルゴリズム（accelerated scan）の使用フラグ
USE_FLEX_ATTN = True              # Flex Attention（柔軟なアテンション実装）の使用フラグ（最新のPyTorch＋CUDAが必要）
USE_FAST_INFERENCE = False         # 推論時にキャッシュ等を利用して高速化するかのフラグ

# wandb experiment tracker

import wandb
wandb.init(project = PROJECT_NAME, mode = 'disabled' if not WANDB_ONLINE else 'online')
wandb.run.name = RUN_NAME
wandb.run.save()

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# memory model

if USE_MEM_ATTENTION_MODEL:
    neural_memory_model = MemoryAttention(
        dim = 64
    )
else:
    neural_memory_model = MemoryMLP(
        dim = 64,
        depth = NEURAL_MEMORY_DEPTH
    )

# instantiate memory-as-context transformer

model = MemoryAsContextTransformer(
    num_tokens = 256,
    dim = 384,
    depth = 8,
    segment_len = WINDOW_SIZE,
    num_persist_mem_tokens = NUM_PERSIST_MEM,
    num_longterm_mem_tokens = NUM_LONGTERM_MEM,
    neural_memory_layers = NEURAL_MEM_LAYERS,
    neural_memory_segment_len = NEURAL_MEM_SEGMENT_LEN,
    neural_memory_batch_size = NEURAL_MEM_BATCH_SIZE,
    neural_mem_gate_attn_output = NEURAL_MEM_GATE_ATTN_OUTPUT,
    neural_mem_weight_residual = NEURAL_MEM_WEIGHT_RESIDUAL,
    neural_memory_qkv_receives_diff_views = NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW,
    use_flex_attn = USE_FLEX_ATTN,
    sliding_window_attn = SLIDING_WINDOWS,
    neural_memory_model = neural_memory_model,
    neural_memory_kwargs = dict(
        dim_head = 64,
        heads = 4,
        attn_pool_chunks = STORE_ATTN_POOL_CHUNKS,
        qk_rmsnorm = NEURAL_MEM_QK_NORM,
        momentum = NEURAL_MEM_MOMENTUM,
        momentum_order = NEURAL_MEM_MOMENTUM_ORDER,
        default_step_transform_max_lr = NEURAL_MEM_MAX_LR,
        use_accelerated_scan = USE_ACCELERATED_SCAN,
        per_parameter_lr_modulation = MEMORY_MODEL_PER_LAYER_LEARNED_LR
    )
).cuda()

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:
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

# optimizer

optim = AdoptAtan2(model.parameters(), lr = LEARNING_RATE)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10., desc = 'training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader), return_loss = True)
        loss.backward()

    print(f'training loss: {loss.item()}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()
    wandb.log(dict(loss = loss.item()))

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader), return_loss = True)
            print(f'validation loss: {loss.item()}')

    if SHOULD_GENERATE and i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        print(f'%s \n\n %s', (prime, '*' * 100))

        sample = model.sample(inp[None, ...], GENERATE_LENGTH, use_cache = USE_FAST_INFERENCE)
        output_str = decode_tokens(sample[0])
        print(output_str)
