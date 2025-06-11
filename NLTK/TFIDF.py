import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.util import ngrams

# Ensure NLTK resources are downloaded
nltk.download('punkt')

# Function to generate n-grams
def generate_ngrams(text, n):
    tokens = nltk.word_tokenize(text)
    n_grams = list(ngrams(tokens, n))
    return [' '.join(gram) for gram in n_grams]

# Function to compute TF-IDF
def compute_tfidf(corpus):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names

# Example usage
if __name__ == "__main__":
    # Sample text corpus
    corpus = [
        "This is a sample document.",
        "This document is another example.",
        "TF-IDF is a useful technique."
    ]

    # Generate n-grams (e.g., bigrams)
    n = 2
    for text in corpus:
        print(f"{n}-grams for '{text}': {generate_ngrams(text, n)}")

    # Compute TF-IDF
    tfidf_matrix, feature_names = compute_tfidf(corpus)
    print("\nTF-IDF Feature Names:")
    print(feature_names)
    print("\nTF-IDF Matrix:")
    print(tfidf_matrix.toarray())