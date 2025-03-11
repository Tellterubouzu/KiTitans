import graphviz

# フローチャートの作成
dot = graphviz.Digraph(comment='MACTransformer Flowchart', format='png')

# ノードの定義
dot.node('A', '入力トークン\n(Input Tokens)')
dot.node('B', '正規化処理\n(Normalization)')
dot.node('C', 'NeuralMemory更新\n(mem.forward)')
dot.node('D', '線形変換\n(Linear Projection)\n→ Q, K, V')
dot.node('E', 'ヘッド分割\n(Split Heads)')
dot.node('F', '永続メモリ結合\n(Concatenate Persistent Memory)')
dot.node('G', 'アテンション計算\n(Attention Computation)')
dot.node('H', 'ヘッド結合\n(Merge Heads)')
dot.node('I', '出力生成\n(Output Generation)')

# ノード間のエッジ（処理順序）の定義
dot.edge('A', 'B', label='Step 1')
dot.edge('B', 'C', label='Step 2')
dot.edge('C', 'D', label='Step 3')
dot.edge('D', 'E', label='Step 4')
dot.edge('E', 'F', label='Step 5')
dot.edge('F', 'G', label='Step 6')
dot.edge('G', 'H', label='Step 7')
dot.edge('H', 'I', label='Step 8')

# フローチャートを表示またはファイルに保存
dot.render('mac_transformer_flowchart', view=True)
