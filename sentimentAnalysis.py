def advanced_sentiment_analysis(documents):
    """
    Uses a Hugging Face Transformers sentiment model (distilbert) for better context-aware analysis.
    
    Args:
        documents (list): A list of strings representing the documents for sentiment analysis.
    
    Returns:
        list: A list of dictionaries with sentiment analysis results for each document. Each result contains:
              - "text": The original document text.
              - "label": The sentiment label ("POSITIVE" or "NEGATIVE").
              - "score": The confidence score for the predicted label.
    """
    if not isinstance(documents, list) or not documents:
        raise ValueError("Must provide a non-empty list of 'documents'.")

    results = []
    for doc in documents:
        # The pipeline returns a list of dicts, each with {label, score}
        pipeline_output = sentiment_pipeline(doc[:512])  # Truncate doc if very long
        # pipeline_output is something like: [{"label": "POSITIVE", "score": 0.99}]
        # We'll just take the first for demonstration
        out = pipeline_output[0]
        results.append({
            "text": doc,
            "label": out["label"],
            "score": out["score"]
        })

    return results
