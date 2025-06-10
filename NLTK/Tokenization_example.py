# import nltk
# nltk.download('punkt')

# from nltk.tokenize import sent_tokenize

# nltk.download('punkt')

# corpus= """NLTK is a leading platform for building Python programs to work with human language data. 
# It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries, and an active discussion forum. NLTK is a great tool for students, educators, and researchers who need to work with human language data.
# It is also widely used in industry for tasks such as text classification, sentiment analysis, and information extraction."""

# # Tokenize the corpus into sentences
# sentences = sent_tokenize(corpus)
# # Print the tokenized sentences
# print(sentences)

# ---------------------------------Above code doesn't work because it is missing the necessary imports and setup for NLTK's Punkt tokenizer. Below is the corrected code that includes the necessary imports and uses the PunktSentenceTokenizer to tokenize sentences from a given text.--------------------------------------------------------------


import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.tokenize import TreebankWordTokenizer

nltk.download('punkt')  # Make sure data is there

text = """NLTK is a leading platform for building Python programs to work with human language data. 
It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries, and an active discussion forum. NLTK is a great tool for students, educators, and researchers who need to work with human language data.
It is also widely used in industry for tasks such as text classification, sentiment analysis, and information extraction."""

# Load English tokenizer from pre-trained punkt data
tokenizer = PunktSentenceTokenizer()

# Tokenize sentences
sentences = tokenizer.tokenize(text)
word_tokenizer = TreebankWordTokenizer()
tokenized_words = [word_tokenizer.tokenize(sentence) for sentence in sentences]
print(sentences)
print(tokenized_words)

