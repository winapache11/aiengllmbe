from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("agentlans/multilingual-e5-small-aligned-sentiment")

def tokenize_text(text):
    return tokenizer(text, return_tensors="pt", padding=True, truncation=True)

def detokenize(tokens):
    return tokenizer.batch_decode(tokens, skip_special_tokens=True)

def input_ids(text):
    tokens = tokenize_text(text)
    return tokens["input_ids"]

if __name__ == "__main__":
    try:
        user_input = input("Enter text to tokenize: ")
    except (EOFError, KeyboardInterrupt):
        print("\nNo input received. Exiting.")
        raise SystemExit(0) 
    
    if not user_input.strip():
        print("No text provided for tokenization.")
        raise SystemExit(0)

    tokens = tokenize_text(user_input)
    detokenized_text = detokenize(tokens["input_ids"])
    input_ids_list = input_ids(user_input)
    print(tokens)
    print("Detokenized text:", detokenized_text)
    print("Input IDs:", input_ids_list)