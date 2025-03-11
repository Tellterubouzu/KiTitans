import os, glob, gzip, json, random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, set_seed
import deepspeed
import wandb

# ===============================
# DeepSpeed 設定（fp16, ZeRO stage2）
# ===============================
ds_config = {
    "train_batch_size": 32,  # DeepSpeed 内部での全バッチサイズ（global batch size）
    "gradient_accumulation_steps": 4,
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "reduce_scatter": True,
        "allgather_bucket_size": 50000000,
        "reduce_bucket_size": 50000000,
        "overlap_comm": True,
        "contiguous_gradients": True
    },
    "steps_per_print": 2000,
    "wall_clock_breakdown": False
}

# ===============================
# WandB の初期化（オンラインモニタリング有効）
# ===============================
PROJECT_NAME = 'titans-mac-transformer'  # プロジェクト名
NUM_LONGTERM_MEM = 8
NEURAL_MEM_LAYERS = (3, 6, 9, 12, 15, 18, 21)
RUN_NAME = f'mac - {NUM_LONGTERM_MEM} longterm mems, layers {NEURAL_MEM_LAYERS}'
WANDB_ONLINE = True  # オンラインログを有効にする
wandb.init(project=PROJECT_NAME, mode='online' if WANDB_ONLINE else 'offline')
wandb.run.name = RUN_NAME
wandb.run.save()

set_seed(123)

# ===============================
# Dataset クラス（複数パターンの jsonl.gz を結合）
# ===============================
class CombinedJsonlGzDataset(Dataset):
    def __init__(self, file_patterns, seq_len, tokenizer):
        """
        file_patterns: glob パターンのリスト。例：
            [
              "/mnt/shimo/Pretrain_corpus/en_wiki/train_*.jsonl.gz",
              "/mnt/shimo/Pretrain_corpus/ja_wiki/train_*.jsonl.gz",
              "/mnt/shimo/Pretrain_corpus/level0/CC-MAIN-2013-2016.jsonl.gz"
            ]
        seq_len: 最大シーケンス長（例: 4096）
        tokenizer: HuggingFace のトークナイザー
        """
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.samples = []
        self.file_list = []
        for pattern in file_patterns:
            files = glob.glob(pattern)
            self.file_list.extend(files)
        print(f"Found {len(self.file_list)} files for patterns: {file_patterns}")
        # 各ファイルの全行を読み込む（大規模データの場合は lazy ローディングに変更を検討）
        for file_path in self.file_list:
            with gzip.open(file_path, 'rt', encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    if "text" in data:
                        self.samples.append(data["text"])
        print(f"Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        enc = self.tokenizer(text, truncation=True, max_length=self.seq_len, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)  # shape: [seq_len]
        return input_ids

# ===============================
# ファイルパスの設定
# ===============================
base_dir = "/mnt/shimo/Pretrain_corpus"

# 学習用：en_wiki の train_*, ja_wiki の train_*、level0/CC-MAIN-2013-2016.jsonl.gz
train_file_patterns = [
    os.path.join(base_dir, "en_wiki", "train_*.jsonl.gz"),
    os.path.join(base_dir, "ja_wiki", "train_*.jsonl.gz"),
    os.path.join(base_dir, "level0", "CC-MAIN-2013-2016.jsonl.gz")
]

# 検証用（英語・日本語それぞれ）
en_val_pattern = os.path.join(base_dir, "en_wiki", "validation_0.jsonl.gz")
ja_val_pattern = os.path.join(base_dir, "ja_wiki", "validation_0.jsonl.gz")

# ===============================
# トークナイザーのロード（sarashina のトークナイザー）
# ===============================
tokenizer = AutoTokenizer.from_pretrained("sbintuitions/sarashina2.2-0.5b-instruct-v0.1")

# ===============================
# Dataset, DataLoader の作成
# ===============================
SEQ_LEN = 4096
train_dataset = CombinedJsonlGzDataset(train_file_patterns, seq_len=SEQ_LEN, tokenizer=tokenizer)
en_val_dataset = CombinedJsonlGzDataset([en_val_pattern], seq_len=SEQ_LEN, tokenizer=tokenizer)
ja_val_dataset = CombinedJsonlGzDataset([ja_val_pattern], seq_len=SEQ_LEN, tokenizer=tokenizer)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
en_val_loader = DataLoader(en_val_dataset, batch_size=4, shuffle=False, num_workers=4, drop_last=True)
ja_val_loader = DataLoader(ja_val_dataset, batch_size=4, shuffle=False, num_workers=4, drop_last=True)

# ===============================
# モデルの初期化（MAC の設定）
# ===============================
from titans_pytorch import MemoryAsContextTransformer, MemoryMLP, MemoryAttention
USE_MEM_ATTENTION_MODEL = False
NEURAL_MEMORY_DEPTH = 2
if USE_MEM_ATTENTION_MODEL:
    neural_memory_model = MemoryAttention(dim=64)
else:
    neural_memory_model = MemoryMLP(dim=64, depth=NEURAL_MEMORY_DEPTH)

model = MemoryAsContextTransformer(
    num_tokens=102400,                # 語彙数 102400
    dim=1120,                         # 隠れ層サイズ 1120
    depth=22,                         # Transformer 層数 22
    segment_len=128,                  # ウィンドウサイズ 128
    num_persist_mem_tokens=8,         # 永続メモリトークン 8
    num_longterm_mem_tokens=8,        # 長期メモリトークン 8
    neural_memory_layers=(3,6,9,12,15,18,21),  # Neural Memory 搭載層
    neural_memory_segment_len=16,     # Neural Memory セグメント長 16
    neural_memory_batch_size=128,
    neural_mem_gate_attn_output=False,
    neural_mem_weight_residual=True,
    neural_memory_qkv_receives_diff_views=True,
    use_flex_attn=True,
    sliding_window_attn=True,
    neural_memory_model=neural_memory_model,
    neural_memory_kwargs=dict(
        dim_head=80,    # 各ヘッドの次元 80 (1120 / 14)
        heads=14,       # Attention ヘッド数 14
        attn_pool_chunks=True,
        qk_rmsnorm=True,
        momentum=True,
        momentum_order=1,
        default_step_transform_max_lr=1e-1,
        use_accelerated_scan=True,
        per_parameter_lr_modulation=True
    ),
    ff_mult=3   # FFN 拡張率 3
).cuda()

# ===============================
# オプティマイザと DeepSpeed の初期化
# ===============================
from adam_atan2_pytorch import AdoptAtan2
optimizer = AdoptAtan2(model.parameters(), lr=2e-4)

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config,
    model_parameters=model.parameters()
)

# ===============================
# 学習ループ
# ===============================
import tqdm
for epoch in range(1):  # epoch0 とする（学習用ファイルすべてを1エポックで走らせる）
    model_engine.train()
    for step, batch in enumerate(train_loader):
        # 各バッチは list of token_id テンソル（各 shape: [seq_len]）なので、pad_sequence を用いて [batch, seq_len] に変換
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0).cuda()
        loss = model_engine(batch, return_loss=True)
        model_engine.backward(loss)
        model_engine.step()
        if step % 100 == 0:
            print(f"Epoch {epoch} Step {step}: Loss = {loss.item()}")
            wandb.log({"train_loss": loss.item(), "step": step})
    # 検証（英語）
    model_engine.eval()
    en_val_losses = []
    with torch.no_grad():
        for batch in en_val_loader:
            batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0).cuda()
            loss = model_engine(batch, return_loss=True)
            en_val_losses.append(loss.item())
    avg_en_val = sum(en_val_losses) / len(en_val_losses)
    print(f"Epoch {epoch} English Validation Loss: {avg_en_val}")
    wandb.log({"en_val_loss": avg_en_val, "epoch": epoch})
    # 検証（日本語）
    ja_val_losses = []
    with torch.no_grad():
        for batch in ja_val_loader:
            batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0).cuda()
            loss = model_engine(batch, return_loss=True)
            ja_val_losses.append(loss.item())
    avg_ja_val = sum(ja_val_losses) / len(ja_val_losses)
    print(f"Epoch {epoch} Japanese Validation Loss: {avg_ja_val}")
    wandb.log({"ja_val_loss": avg_ja_val, "epoch": epoch})
    # チェックポイント保存
    model_engine.save_checkpoint("checkpoints", tag=f"epoch_{epoch}")
