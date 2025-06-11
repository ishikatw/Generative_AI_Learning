from gensim.models import Word2Vec

if __name__ == "__main__":
    # Example usage
    corpus = ["I", "love", "coding", "I", "love", "Python"]
    sentences = [corpus]  # Gensim expects a list of tokenized sentences

    # Train Word2Vec model
    embedding_dim = 10
    model = Word2Vec(sentences, vector_size=embedding_dim, window=5, min_count=1, workers=4)

    # Access word vectors
    print("Vector for 'love':", model.wv['love'])
    print("Vocabulary:", model.wv.index_to_key)

    # Calculate similarity between words
    similarity = model.wv.similarity('love', 'coding')
    print("Similarity between 'love' and 'coding':", similarity)

    # Save and load the model
    model.save("word2vec.model")
    loaded_model = Word2Vec.load("word2vec.model")
    print("Loaded model vector for 'love':", loaded_model.wv['love'])