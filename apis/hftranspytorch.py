from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def class_label(sentence, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english") -> str:
    model_name = model_name.strip()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
    except Exception as e:
        raise ValueError(f"Failed to load model '{model_name}': {e}")
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
    print(f"Input IDs: {inputs['input_ids']}")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax(dim=-1).item()
    
    print(f"Predicted class ID: {predicted_class_id}")

    tokenizer.save_pretrained("../model_directory")
    model.save_pretrained("../model_directory")
    
    return model.config.id2label[predicted_class_id]

def create_model(model_name: str, model_directory: str): 
    model_directory = model_directory.strip()
    try:
        my_tokenizer = AutoTokenizer.from_pretrained(model_directory)
        model_name = AutoModelForSequenceClassification.from_pretrained(model_directory)
        model_name
        return my_tokenizer, model_name
    except Exception as e:
        raise ValueError(f"Failed to create model '{model_name}': {e}")


def save_model(model_name: str, save_directory: str):
    model_name = model_name.strip()
    save_directory = save_directory.strip()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
    except Exception as e:
        raise ValueError(f"Failed to save model '{model_name}' to '{save_directory}': {e}")


if __name__ == "__main__":
    try:
        user_input = input("Enter text for classification: ")
    except (EOFError, KeyboardInterrupt):
        print("\nNo input received. Exiting.")
        raise SystemExit(0)

    if not user_input.strip():
        print("No text provided for classification.")
        raise SystemExit(0)

    label = class_label(user_input)
    print(f"Predicted class label: {label}")

    save_model("distilbert-base-uncased-finetuned-sst-2-english", "../saved_model")
    create_model("my_model", "../saved_model")
    