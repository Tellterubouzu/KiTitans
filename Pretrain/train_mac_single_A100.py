import os
import glob
import gzip
import json
import random
import torch
###################
import torch._dynamo
torch._dynamo.config.suppress_errors = True

####################

from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer, set_seed
import wandb


# -----------------------------
# WandB の初期化（オンラインログ有効）
# -----------------------------
PROJECT_NAME = 'titans-mac-transformer'
NUM_LONGTERM_MEM = 8
NEURAL_MEM_LAYERS = (3, 6, 9, 12, 15, 18, 21)
RUN_NAME = f'mac - {NUM_LONGTERM_MEM} longterm mems, layers {NEURAL_MEM_LAYERS}'
WANDB_ONLINE = True
wandb.init(project=PROJECT_NAME, mode='online' if WANDB_ONLINE else 'offline')
wandb.run.name = RUN_NAME
wandb.run.save()

set_seed(123)

# -----------------------------
# Streaming 用 IterableDataset
# -----------------------------
class StreamingJsonlGzDataset(IterableDataset):
    def __init__(self, file_patterns, seq_len, tokenizer):
        """
        file_patterns: glob パターンのリスト
          例：
            [
              "/mnt/shimo/Pretrain_corpus/en_wiki/train_*.jsonl.gz",
              "/mnt/shimo/Pretrain_corpus/ja_wiki/train_*.jsonl.gz",
              "/mnt/shimo/Pretrain_corpus/level0/CC-MAIN-2013-2016.jsonl.gz"
            ]
        seq_len: 最大シーケンス長（例: 4096）
        tokenizer: HuggingFace のトークナイザー
        """
        self.file_patterns = file_patterns
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.file_list = []
        for pattern in file_patterns:
            files = glob.glob(pattern)
            self.file_list.extend(files)
        print(f"Found {len(self.file_list)} files for patterns: {file_patterns}")

    def __iter__(self):
        for file_path in self.file_list:
            with gzip.open(file_path, 'rt', encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "text" in data:
                            text = data["text"]
                            enc = self.tokenizer(text, truncation=True, max_length=self.seq_len, return_tensors="pt")
                            yield enc["input_ids"].squeeze(0)
                    except Exception as e:
                        continue

# -----------------------------
# ファイルパス設定
# -----------------------------
base_dir = "/mnt/shimo/Pretrain_corpus"
train_file_patterns = [
    os.path.join(base_dir, "en_wiki", "train_*.jsonl.gz"),
    os.path.join(base_dir, "ja_wiki", "train_*.jsonl.gz"),
    os.path.join(base_dir, "level0", "CC-MAIN-2013-2016.jsonl.gz")
]
en_val_pattern = os.path.join(base_dir, "en_wiki", "validation_0.jsonl.gz")
ja_val_pattern = os.path.join(base_dir, "ja_wiki", "validation_0.jsonl.gz")

# -----------------------------
# トークナイザーのロード
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("sbintuitions/sarashina2.2-0.5b-instruct-v0.1")

# -----------------------------
# Dataset, DataLoader の作成
# -----------------------------
SEQ_LEN = 2048
train_dataset = StreamingJsonlGzDataset(train_file_patterns, seq_len=SEQ_LEN, tokenizer=tokenizer)
en_val_dataset = StreamingJsonlGzDataset([en_val_pattern], seq_len=SEQ_LEN, tokenizer=tokenizer)
ja_val_dataset = StreamingJsonlGzDataset([ja_val_pattern], seq_len=SEQ_LEN, tokenizer=tokenizer)


# collate関数を定義
def pad_collate_fn(batch, pad_token_id=0):
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_token_id)
    return batch
# トークナイザーからパディングトークンを取得
pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

# DataLoaderの定義（collate_fnを指定）
train_loader = DataLoader(
train_dataset,batch_size=4,num_workers=4,collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id))
en_val_loader = DataLoader(en_val_dataset,batch_size=4,num_workers=4,collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id))
ja_val_loader = DataLoader(ja_val_dataset,batch_size=4,num_workers=4,collate_fn=lambda batch: pad_collate_fn(batch, pad_token_id))


# -----------------------------
# モデル初期化（MAC の設定）
# -----------------------------
from titans_pytorch import MemoryAsContextTransformer, MemoryMLP, MemoryAttention
USE_MEM_ATTENTION_MODEL = False
NEURAL_MEMORY_DEPTH = 2
if USE_MEM_ATTENTION_MODEL:
    neural_memory_model = MemoryAttention(dim=80)
else:
    neural_memory_model = MemoryMLP(dim=80, depth=NEURAL_MEMORY_DEPTH)

model = MemoryAsContextTransformer(
    num_tokens=102400,
    dim=1120,
    depth=22,
    segment_len=128,
    num_persist_mem_tokens=8,
    num_longterm_mem_tokens=8,
    neural_memory_layers=(3,6,9,12,15,18,21),
    neural_memory_segment_len=16,
    neural_memory_batch_size=128,
    neural_mem_gate_attn_output=False,
    neural_mem_weight_residual=True,
    neural_memory_qkv_receives_diff_views=True,
    use_flex_attn=False,
    #use_flex_attn=True,
    sliding_window_attn=True,
    neural_memory_model=neural_memory_model,
    neural_memory_kwargs=dict(
        dim_head=80,
        heads=14,
        attn_pool_chunks=True,
        qk_rmsnorm=True,
        momentum=True,
        momentum_order=1,
        default_step_transform_max_lr=1e-1,
        use_accelerated_scan=True,
        per_parameter_lr_modulation=True
    ),
    ff_mult=3
).cuda()

# -----------------------------
# オプティマイザの設定
# -----------------------------
from adam_atan2_pytorch import AdoptAtan2
optimizer = AdoptAtan2(model.parameters(), lr=2e-4)

# -----------------------------
# チェックポイント保存設定
# -----------------------------
checkpoint_dir = "/home/shimomura/Devenv/checkpoint/"
os.makedirs(checkpoint_dir, exist_ok=True)

def save_checkpoint(step, epoch):
    ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch}_step_{step}.pt")
    torch.save(model.state_dict(), ckpt_path)
    # チェックポイントが10個を超えたら、古いものから削除
    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt")), key=os.path.getmtime)
    if len(ckpt_files) > 10:
        files_to_remove = ckpt_files[:len(ckpt_files) - 10]
        for f in files_to_remove:
            os.remove(f)
    print(f"Checkpoint saved at step {step}, epoch {epoch}")

# -----------------------------
# 学習ループ（単一GPU）
# -----------------------------
num_epochs = 1  # epoch0 とする
grad_accum_steps = 8  # バッチサイズ4×8 = 実質32
model.train()
step_count = 0
for epoch in range(num_epochs):
    for batch in train_loader:
        # 各サンプルは [seq_len] テンソルなので、pad_sequence で [batch, seq_len] に変換
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0).cuda()
        loss = model(batch, return_loss=True)
        loss.backward()
        step_count += 1
        if step_count % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()
        if step_count % 100 == 0:
            print(f"Epoch {epoch} Step {step_count}: Loss = {loss.item()}")
            wandb.log({"train_loss": loss.item(), "step": step_count, "epoch": epoch})
        # 3000ステップごとにチェックポイント保存
        if step_count % 3000 == 0:
            save_checkpoint(step_count, epoch)
    # -----------------------------
    # 検証ループ（英語）
    # -----------------------------
    model.eval()
    en_val_losses = []
    with torch.no_grad():
        for batch in en_val_loader:
            batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0).cuda()
            loss = model(batch, return_loss=True)
            en_val_losses.append(loss.item())
    avg_en_val = sum(en_val_losses) / len(en_val_losses) if en_val_losses else 0.0
    print(f"Epoch {epoch} English Validation Loss: {avg_en_val}")
    wandb.log({"en_val_loss": avg_en_val, "epoch": epoch})
    # -----------------------------
    # 検証ループ（日本語）
    # -----------------------------
    ja_val_losses = []
    with torch.no_grad():
        for batch in ja_val_loader:
            batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0).cuda()
            loss = model(batch, return_loss=True)
            ja_val_losses.append(loss.item())
    avg_ja_val = sum(ja_val_losses) / len(ja_val_losses) if ja_val_losses else 0.0
    print(f"Epoch {epoch} Japanese Validation Loss: {avg_ja_val}")
    wandb.log({"ja_val_loss": avg_ja_val, "epoch": epoch})
    # -----------------------------
    # エポック終了時のチェックポイント保存（必要なら）
    # -----------------------------
    save_checkpoint(step_count, epoch)
    model.train()
