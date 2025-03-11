print("最初の10件のデータ:")
for i, data in enumerate(train_dataset):
    print(f"データ {i+1}: {data}")
    if i >= 9:
        break
