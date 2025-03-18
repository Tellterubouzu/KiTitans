#!/usr/bin/env python
import glob
import gzip
import random
import os

def collect_lines_from_files(file_list):
    """
    指定されたgzファイル群から全行を読み込み、リストとして返す関数です。
    ※ 各行が1サンプルである前提です。必要に応じてパース処理を追加してください。
    """
    all_lines = []
    for file_path in file_list:
        print(f"Processing {file_path} ...")
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                # 各行を読み込み、改行文字を除去してリストに追加
                lines = [line.strip() for line in f if line.strip()]
                all_lines.extend(lines)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return all_lines

def write_lines_to_gz(lines, output_path):
    """
    行のリストを指定された出力パスに gz 圧縮形式で保存します。
    """
    with gzip.open(output_path, 'wt', encoding='utf-8') as f:
        for line in lines:
            f.write(line + "\n")
    print(f"Saved {len(lines)} lines to {output_path}")

def main():
    # 出力先ディレクトリ（存在しなければ作成）
    output_dir = "./epoch_datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    # Epoch0 用ファイルリストの作成
    en_wiki_files = glob.glob("/mnt/shimo/Pretrain_corpus/en_wiki/train_*.jsonl.gz")
    ja_wiki_files = glob.glob("/mnt/shimo/Pretrain_corpus/ja_wiki/train_*.jsonl.gz")
    level0_files_epoch0 = glob.glob("/mnt/shimo/Pretrain_corpus/level0/CC-MAIN-2013-2016*.jsonl.gz")
    epoch0_files = en_wiki_files + ja_wiki_files + level0_files_epoch0
    
    # # Epoch1 用ファイルリストの作成
    # c4_files = glob.glob("/mnt/shimo/Pretrain_corpus/c4/*.jsonl.gz")
    # # level0全体から、Epoch0で使用したファイルを除外
    # all_level0_files = glob.glob("/mnt/shimo/Pretrain_corpus/level0/CC-MAIN-*.jsonl.gz")
    # level0_files_epoch1 = [f for f in all_level0_files if f not in level0_files_epoch0]
    # epoch1_files = c4_files + level0_files_epoch1

    # print("Epoch0 対象ファイル数:", len(epoch0_files))
    # print("Epoch1 対象ファイル数:", len(epoch1_files))
    
    # 各Epochのデータセットを作成し、保存
    for epoch, file_list in [(0, epoch0_files)]:#, (1, epoch1_files)]:
        print(f"\nEpoch {epoch} のデータを作成中...")
        lines = collect_lines_from_files(file_list)
        print(f"Epoch {epoch}: 合計 {len(lines)} 行を収集しました。")
        random.shuffle(lines)
        output_path = os.path.join(output_dir, f"epoch{epoch}_dataset.jsonl.gz")
        write_lines_to_gz(lines, output_path)
        print(f"Epoch {epoch} のデータセットを {output_path} に保存しました。")

if __name__ == "__main__":
    main()
