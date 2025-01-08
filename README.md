# Instructions for Setting Up and Running Sentiment Analysis

This document provides step-by-step instructions to set up and run a sentiment analysis function using the Hugging Face Transformers library, PyTorch, and spaCy.

---

## 1. Prerequisites

Ensure that the following tools are installed on your system:

1. **Python** (version 3.7 or higher)
2. **pip** (Python package manager)

---

## 2. Setting Up the Environment

### Create a Virtual Environment (Optional but Recommended)

1. Create a virtual environment:
   ```bash
   python -m venv sentiment_env
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     sentiment_env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source sentiment_env/bin/activate
     ```

### Install Dependencies

1. Save the following `requirements.txt` file:

   ```plaintext
   transformers==4.34.0          # Hugging Face Transformers for sentiment analysis
   torch==2.0.1                  # PyTorch backend required by Transformers
   spacy==3.7.0                  # NLP preprocessing library
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the `en_core_web_sm` model for spaCy:
   ```bash
   python -m spacy download en_core_web_sm
   ```

---

## 3. Using the Sentiment Analysis Function

### Function Code
Save the following code into a Python file (e.g., `sentiment_analysis.py`):

```python
from transformers import pipeline
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize the Hugging Face sentiment analysis pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def advanced_sentiment_analysis(documents):
    """
    Perform sentiment analysis on a list of documents.

    Args:
        documents (list): A list of strings representing the documents.

    Returns:
        list: A list of dictionaries containing the sentiment label and confidence score.
    """
    if not isinstance(documents, list) or not documents:
        raise ValueError("Must provide a non-empty list of 'documents'.")

    results = []
    for doc in documents:
        # Use the pipeline to analyze sentiment
        pipeline_output = sentiment_pipeline(doc[:512])  # Truncate long texts
        out = pipeline_output[0]  # Get the first result
        results.append({
            "text": doc,
            "label": out["label"],
            "score": out["score"]
        })

    return results

# Example usage
if __name__ == "__main__":
    documents = [
        "The product was excellent, I absolutely loved it!",
        "I am disappointed with the service, it was terrible."
    ]
    analysis_results = advanced_sentiment_analysis(documents)
    for result in analysis_results:
        print(f"Text: {result['text']}\nLabel: {result['label']}\nScore: {result['score']}\n")
```

---

## 4. Running the Sentiment Analysis

1. Run the Python script:
   ```bash
   python sentiment_analysis.py
   ```

2. The output should display the sentiment label (e.g., POSITIVE or NEGATIVE) and the confidence score for each document provided.

Example output:
```plaintext
Text: The product was excellent, I absolutely loved it!
Label: POSITIVE
Score: 0.9987

Text: I am disappointed with the service, it was terrible.
Label: NEGATIVE
Score: 0.9932
```

---

## 5. Troubleshooting

### Common Issues:
1. **"Command not found" errors**:
   Ensure Python and pip are installed and added to your system's PATH.

2. **Library installation issues**:
   If dependencies fail to install, ensure you are connected to the internet and have the correct Python version.

3. **spaCy language model not found**:
   Run:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Long documents causing truncation**:
   The function truncates documents to 512 characters for compatibility with the Transformer model. Consider summarizing longer texts before analysis.

---

## 6. Customization

You can replace the `distilbert-base-uncased-finetuned-sst-2-english` model with other sentiment analysis models available on the Hugging Face Model Hub. For example:

1. Visit [Hugging Face Model Hub](https://huggingface.co/models).
2. Search for sentiment analysis models.
3. Update the `pipeline` initialization in the code:
   ```python
   sentiment_pipeline = pipeline("sentiment-analysis", model="<model-name>")
   ```
