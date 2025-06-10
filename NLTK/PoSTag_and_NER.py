import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk import pos_tag, ne_chunk

# Ensure required resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

text = """NLTK is a leading platform for building Python programs to work with human language data. 
It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries, and an active discussion forum. NLTK is a great tool for students, educators, and researchers who need to work with human language data.
It is also widely used in industry for tasks such as text classification, sentiment analysis, and information extraction."""

# Load English tokenizer from pre-trained punkt data
tokenizer = PunktSentenceTokenizer()

# Tokenize sentences
sentences = tokenizer.tokenize(text)
word_tokenizer = TreebankWordTokenizer()
tokenized_words = [word_tokenizer.tokenize(sentence) for sentence in sentences]

# Perform POS tagging
pos_tags = [pos_tag(word) for word in tokenized_words]

# Perform Named Entity Recognition
named_entities = [ne_chunk(pos_tag(word)) for word in tokenized_words]

# Print results
print("Sentences:")
print(sentences)
print("\nTokenized Words:")
print(tokenized_words)
print("\nPOS Tags:")
print(pos_tags)
print("\nNamed Entities:")
for entity in named_entities:
    print(entity)