from transformers import pipeline

# Try to create the pipeline, but handle the case where no ML backend is installed
try:
    sentiment_classifier = pipeline("sentiment-analysis", model="agentlans/multilingual-e5-small-aligned-sentiment")
except NameError as e:
    # This typically happens when transformers expects 'torch' (or another backend) but it's not available
    print("Backend ML library not found. Install PyTorch or TensorFlow (>=2.0) to enable model inference.")
    print("Error:", e)
    sentiment_classifier = None


if __name__ == "__main__":
    try:
        user_input = input("Enter text for sentiment analysis: ")
    except (EOFError, KeyboardInterrupt):
        print("\nNo input received. Exiting.")
        raise SystemExit(0)

    if not user_input.strip():
        print("No text provided for analysis.")
        raise SystemExit(0)

    if sentiment_classifier is None:
        print("Cannot run sentiment analysis because no ML backend is installed.\nSuggested fix:\n  1) Activate the project's venv: source ../venv/bin/activate\n  2) Install PyTorch (CPU): pip install --index-url https://download.pytorch.org/whl/cpu torch\nOr install the appropriate build for your mac (MPS) following https://pytorch.org/get-started/locally/")
        raise SystemExit(1)

    result = sentiment_classifier(user_input)
    print(f"Sentiment: {result[0]['label']}, Score: {result[0]['score']}")