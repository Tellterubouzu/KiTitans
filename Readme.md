# KiTAI-KiTitans Project
Kyusyu Institute of Technologyで勝手にやってるTitansをつかった超小型の雑談ようのLLMをつくってみようの会 のリポジトリ
# 情報
モデルサイズは1B以下でエッジデバイスで動くレベル
ベンチマーク性能というよりは自然な会話・雑談ができるようにする
データセットCommonCrawlとllm-jp,swallow,Tanuki,qwenが使ってるやつ+蒸留？
とりあえず実装してみる

## 実装
optimizer →どうせならMoun optimizerやってみる？

### 進捗
環境構築終了 (03.10)
データセット準備



## アーキテクチャ :Memory as Context Transformer
* 語彙数 : 65024
* 隠れ層の次元:1024
* ヘッドの数:16 (ヘッドの次元は1025/16=64)
* 全層数:24
* Neural Mem layers = (2,4,...,24(で12層にMemoryMLP))
* Neural Memory MLP depth=2(拡大率2)

## パラメータ数
【埋め込み層・出力層等】
* トークン埋め込み層：65,024 × 1,024 ≒ 66.6M
* 出力層（to_logits）：1,024 × 65,024 ≒ 66.6M
* 位置エンベディングなど（概ね）：約2M
<br>→ 小計：約66.6M + 66.6M + 2M ≒ 135M

【各 Transformer 層（全24層）】<br>
(1) 【Self-Attention ブロック】<br>
* QKV投影：Linear(1,024, 3×(16×64)=3,072) → 1,024×3,072 = 3,145,728
*  出力変換：Linear(1,024, 1,024) → 1,024×1,024 = 1,048,576
*  Persistent Memory（各層で、例えば形状が (2, 16, 16, 64) と仮定）→ 約32,768
<br>→ 1層あたり約 3,145,728 + 1,048,576 + 32,768 ≒ 4,227,072
<br>→ 24層で：4,227,072 × 24 ≒ 101M

(2) 【FeedForward ブロック】
*  一般的に隠れ次元は4倍と仮定（1,024→4,096→1,024）
* 1層あたり：1,024×4,096 + 4,096×1,024 = 8,388,608
<br>→ 24層で：8,388,608 × 24 ≒ 201M

(3) 【ハイパーコネクション等】<br>
各層での補助的な接続部（init_hyper_conn や mem_hyper_conn、mem_qkv_layer_selector など）を全体で約66Mと見積もる
<br>
→ Transformer 層全体での合計は：101M + 201M + 66M ≒ 368M
<br>
【神経記憶モジュール（MemoryMLP）
<br>USE_MEM_ATTENTION_MODEL=False の場合、MemoryMLP を使用します。MemoryMLP の内訳は：
* 入力から第1層：64 × 128 = 8,192
* 第2層：128 × 64 = 8,192
<br>→ 1モジュールあたり合計：約16,384 (約0.016M)<br>
適用は12層なので：16,384 × 12 ≒ 196,608 ≒ 0.2M<br>

【総計】<br>
埋め込み・出力関連：約135M<br>
Transformer 層（Self-Attention＋FeedForward＋ハイパーコネクション）：約368M<br>
神経記憶モジュール（MemoryMLP）：約0.2M<br>
これらを合計すると：<br>
135M + 368M + 0.2M ≒ 503M
