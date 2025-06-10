import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

# Create a WordNetLemmatizer object
lemmatizer = WordNetLemmatizer()
# Example words to lemmatize
words = ["running", "ran", "easily", "fairly", "eating", "eaten", "eats"]
# Lemmatize the words
lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word in words]
# Print the lemmatized words
print("WordNet Lemmatizer Example")
print("Original words:", words)
print("Lemmatized words:", lemmatized_words)

# Output:
# Original words: ['running', 'ran', 'easily', 'fairly', 'eating', 'eaten', 'eats']
# Lemmatized words: ['run', 'run', 'easily', 'fairly', 'eat', 'eat', 'eat']

# The WordNetLemmatizer uses the WordNet lexical database to lemmatize words.
# It reduces words to their base or dictionary form, which is useful for natural language processing tasks.
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# The lemmatization process is more sophisticated than stemming, as it considers the context and part of speech of the word.
# This allows it to produce more accurate base forms of words, especially for verbs and nouns.