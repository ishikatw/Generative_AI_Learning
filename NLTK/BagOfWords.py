from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from sklearn.feature_extraction.text import CountVectorizer

# Ensure NLTK resources are downloaded
nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocess the input text by converting to lowercase, removing non-alphanumeric characters,
    removing stopwords, and applying Porter stemming.
    """
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    # Apply Porter stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    
    return ' '.join(stemmed_words)

def bag_of_words(text):
    """
    Generate a bag of words representation for the given text.
    """
    words = text.split()
    word_counts = Counter(words)
    return word_counts

def bag_of_words_vector(texts):
    """
    Generate a Bag of Words vector representation for a list of texts.
    """
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(texts)
    return bow_matrix, vectorizer.get_feature_names_out()

if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "This is a sample text. Text processing is fun!",
        "Another example of text processing.",
        "Text analysis is interesting."
    ]
    
    # Preprocess each text
    processed_texts = [preprocess_text(text) for text in sample_texts]
    
    # Generate Bag of Words dictionary
    bow_dict = [bag_of_words(text) for text in processed_texts]
    print("Bag of Words Dictionary:", bow_dict)
    
    # Generate Bag of Words vector
    bow_matrix, feature_names = bag_of_words_vector(processed_texts)
    print("\nBag of Words Vector (Sparse Matrix):\n", bow_matrix.toarray())
    print("\nFeature Names:", feature_names)