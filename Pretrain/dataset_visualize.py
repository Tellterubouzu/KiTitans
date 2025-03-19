import gzip
import json
import pandas as pd

# ファイルパスの指定
file_path = "./epoch_datasets/epoch0_dataset.jsonl.gz"

# 最初の30行分のデータを格納するリスト
data = []

# gzip形式のJSONLファイルをテキストモードでオープンして読み込む
with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 30:
            break
        # 各行のJSON文字列を辞書型に変換してリストに追加
        data.append(json.loads(line))

# データをPandas DataFrameに変換
df = pd.DataFrame(data)

# CSVファイルに出力（ヘッダー付き・インデックスなし）
output_csv = "epoch0_dataset_first30.csv"
df.to_csv(output_csv, index=False, encoding='utf-8-sig')

print(f"CSVファイル '{output_csv}' に変換しました。")
