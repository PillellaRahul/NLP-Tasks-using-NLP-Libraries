# NLP Tasks - Quick Guide

This repository provides simple examples of fundamental NLP tasks using popular Python libraries. Below are the tasks along with questions to help you understand the code.

---

## Tokenization
### Code
```python
from nltk.tokenize import word_tokenize
text = "I love NLP. It's amazing!"
print(word_tokenize(text))
```
### Question
- How are sentences split into tokens in this example?

---

## Stemming
### Code
```python
from nltk.stem import PorterStemmer
ps = PorterStemmer()
print(ps.stem("running"))
```
### Question
- What effect does stemming have on inflected words?

---

## Lemmatization
### Code
```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("running", pos='v'))
```
### Question
- How does lemmatization handle different parts of speech?

---

## Stop Word Removal
### Code
```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

text = "This is a simple NLP example."
stop_words = set(stopwords.words('english'))
words = word_tokenize(text)
filtered_words = [word for word in words if word.lower() not in stop_words]
print("Filtered Words:", filtered_words)
```
### Question
- Why is it important to remove stop words in NLP tasks?

---

## Part-of-Speech (POS) Tagging
### Code
```python
from nltk import pos_tag
from nltk.tokenize import word_tokenize

text = "I love coding."
words = word_tokenize(text)
pos_tags = pos_tag(words)
print("POS Tags:", pos_tags)
```
### Question
- How do POS tags help in understanding the grammatical structure of a sentence?

---

## Named Entity Recognition (NER)
### Code
```python
import spacy

nlp = spacy.load("en_core_web_sm")
text = "Barack Obama was the president of the United States."
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)
```
### Question
- What types of entities are identified in the given text?

---

## Dependency Parsing
### Code
```python
import spacy

nlp = spacy.load("en_core_web_sm")
text = "I love NLP."
doc = nlp(text)

for token in doc:
    print(token.text, "\u2192", token.dep_, "\u2192", token.head.text)
```
### Question
- How does dependency parsing represent the relationships between words?

---

## Bag-of-Words (BoW)
### Code
```python
from sklearn.feature_extraction.text import CountVectorizer

texts = ["I love NLP", "NLP is fun"]
vectorizer = CountVectorizer()
bow = vectorizer.fit_transform(texts)
print("Vocabulary:", vectorizer.vocabulary_)
print("Bag-of-Words Matrix:\n", bow.toarray())
```
### Question
- What information does the Bag-of-Words representation capture about the text?

---

## TF-IDF
### Code
```python
from sklearn.feature_extraction.text import TfidfVectorizer

texts = ["I love NLP", "NLP is fun"]
tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(texts)
print("TF-IDF Matrix:\n", tfidf.toarray())
```
### Question
- How does TF-IDF weigh the importance of terms in a document?

---

## Word Embeddings
### Code
```python
from gensim.models import Word2Vec
sentences = [["I", "love", "NLP"], ["NLP", "is", "fun"]]
model = Word2Vec(sentences, vector_size=5, min_count=1)
print(model.wv['NLP'])
```
### Question
- How do word embeddings capture semantic relationships between words?

---

## Transformers
### Code
```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
print(classifier("I love learning NLP!"))
```
### Question
- What advantages do transformer-based models like BERT offer over traditional methods?

---

## How to Run
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Explore the examples and answer the questions!

---

## Contributing
Feel free to enhance the examples or add new questions.

---

## License
This repository is licensed under the MIT License.
