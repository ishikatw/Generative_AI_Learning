from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Get English stopwords
stop_words = set(stopwords.words('english'))
# Example sentence
sentence = "This is an example sentence for stopword removal. It contains some common words that should be filtered out. "
# Tokenize the sentence into words
words_in_sentence = sentence.split()
# Remove stopwords from the sentence
filtered_words = [word for word in words_in_sentence if word.lower() not in stop_words]
# Print the filtered words
print("Original sentence:", sentence)
print("Filtered words:", filtered_words)

# The stopwords module provides a list of common words that are often removed from text during preprocessing.
# These words do not carry significant meaning and can be safely ignored in many text processing tasks.
# The filtered words are the remaining words after removing the stopwords.