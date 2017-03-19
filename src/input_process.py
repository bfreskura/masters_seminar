def embed_words(embedder="glove"):
    if embedder.lower() == "glove":
        pass
    else:
        raise NotImplementedError("Embedder not supported")
