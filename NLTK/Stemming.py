from nltk.stem import PorterStemmer

# Create a PorterStemmer object
stemmer = PorterStemmer()
# Example words to stem
words = ["running", "ran", "easily", "fairly","eating", "eaten", "eats"]
# Stem the words
stemmed_words = [stemmer.stem(word) for word in words]
# Print the stemmed words
print("PorterStemmer Example")
print("Original words:", words)
print("Stemmed words:", stemmed_words)


# Output:
# Original words: ['running', 'ran', 'easily', 'fairly']
# Stemmed words: ['run', 'ran', 'easili', 'fairli']

# The PorterStemmer is a widely used stemming algorithm that reduces words to their root form.
# It is particularly effective for English words and is commonly used in natural language processing tasks.

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from nltk.stem import RegexpStemmer
# Create a RegexpStemmer object with a custom pattern
regexp_stemmer = RegexpStemmer('ing$|ed$|s$', min=4)
# Example words to stem
words = ["running", "ran", "easily", "fairly", "eating", "eaten", "eats"]
# Stem the words using the RegexpStemmer
stemmed_words = [regexp_stemmer.stem(word) for word in words]
# Print the stemmed words
print("Regexp Stemmer Example")
print("Original words:", words)
print("Stemmed words:", stemmed_words)


# Output:
# Original words: ['running', 'ran', 'easily', 'fairly', 'eating', 'eaten', 'eats']
# Stemmed words: ['run', 'ran', 'easili', 'fairli', 'eat', 'eat', 'eat']

# The RegexpStemmer allows you to define custom patterns for stemming words.
# In this example, it removes the suffixes 'ing' and 'ed' from words that are at least 4 characters long.

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from nltk.stem import SnowballStemmer
# Create a SnowballStemmer object for English
snowball_stemmer = SnowballStemmer("english")
# Example words to stem 
words = ["running", "ran", "easily", "fairly", "eating", "eaten", "eats"]
# Stem the words using the SnowballStemmer
stemmed_words = [snowball_stemmer.stem(word) for word in words]
# Print the stemmed words
print("Snowball Stemmer Example")
print("Original words:", words)
print("Stemmed words:", stemmed_words) 


# Output:
# Original words: ['running', 'ran', 'easily', 'fairly', 'eating', 'eaten', 'eats']
# Stemmed words: ['run', 'ran', 'easili', 'fairli', 'eat', 'eat', 'eat']

# The SnowballStemmer is a more advanced stemming algorithm that supports multiple languages.
# It is particularly effective for English and provides better stemming results compared to the PorterStemmer.
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------