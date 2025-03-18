from transformers import AutoTokenizer, AutoConfig

def main(model_name):
    
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
    main(model_name="Qwen/Qwen2-0.5B")
    main(model_name="weblab-GENIAC/Tanuki-8B-dpo-v1.0")
    main(model_name = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1")
    main(model_name = "llm-jp/llm-jp-3-440m-instruct2")
'''
=== モデル設定情報 ===
モデル名: Qwen/Qwen2-0.5B
語彙サイズ: 151936
隠れ層の次元 (hidden_size): 896
層の数 (num_hidden_layers): 24
Attentionヘッドの数 (num_attention_heads): 14
=== モデル設定情報 ===
モデル名: weblab-GENIAC/Tanuki-8B-dpo-v1.0
語彙サイズ: 65024
隠れ層の次元 (hidden_size): 4096
層の数 (num_hidden_layers): 32
Attentionヘッドの数 (num_attention_heads): 32
=== モデル設定情報 ===
モデル名: sbintuitions/sarashina2.2-0.5b-instruct-v0.1
語彙サイズ: 102400
隠れ層の次元 (hidden_size): 1280
層の数 (num_hidden_layers): 24
Attentionヘッドの数 (num_attention_heads): 16
=== モデル設定情報 ===
モデル名: llm-jp/llm-jp-3-440m-instruct2
語彙サイズ: 99584
隠れ層の次元 (hidden_size): 1024
層の数 (num_hidden_layers): 16
Attentionヘッドの数 (num_attention_heads): 8
'''