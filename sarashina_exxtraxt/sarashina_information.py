from transformers import AutoTokenizer, AutoConfig

def main():
    model_name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"
    
    # トークナイザーのロード（日本語モデルとしてそのまま利用）
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # モデル設定のロード
    config = AutoConfig.from_pretrained(model_name)
    
    print("=== モデル設定情報 ===")
    print(f"モデル名: {model_name}")
    print(f"語彙サイズ: {config.vocab_size}")
    print(f"隠れ層の次元 (hidden_size): {config.hidden_size}")
    print(f"層の数 (num_hidden_layers): {config.num_hidden_layers}")
    print(f"Attentionヘッドの数 (num_attention_heads): {config.num_attention_heads}")

if __name__ == "__main__":
    main()
'''
=== モデル設定情報 ===
モデル名: sbintuitions/sarashina2.2-0.5b-instruct-v0.1
語彙サイズ: 102400
隠れ層の次元 (hidden_size): 1280
層の数 (num_hidden_layers): 24
Attentionヘッドの数 (num_attention_heads): 16
'''