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
* ウィンドウサイズ128
* Neueal Memory セグメント長さ16
* 永続メモリトークン8
* 長期メモリトークン9
* 語彙数:102400
* 隠れ層サイズ:1120
* Transformer層数22層
* Attentionヘッド14
* FFN拡張率3
* Neural Memory module:3,6,9,12,15,18,21

## Kititans Flow -Chart

flowchart TD
    A[入力トークンシーケンス]
    B[トークン埋め込み]
    C[位置エンコーディング追加]
    D[永続メモリトークンの初期化]
    E[Transformer層ループ開始]
    F[現在の層はNeuralMemory更新対象か？]
    G[NeuralMemory更新 (mem.forward)]
    H[クエリ・キー・バリュープロジェクション]
    I[アテンション計算\n(スライディングウィンドウ or Flex Attention)]
    J[ヘッド統合 & 残差接続]
    K[フィードフォワードネットワーク]
    L[次層へループ/最終出力]
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F -- Yes --> G
    F -- No --> H
    G --> H
    H --> I
    I --> J
    J --> K
    K --> E
    E --> L

# お家に帰るのだ