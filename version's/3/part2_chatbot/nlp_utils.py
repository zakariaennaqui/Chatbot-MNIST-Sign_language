import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def lemmatize(words):
    return [lemmatizer.lemmatize(w.lower()) for w in words]

def bag_of_words(sentence, vocabulary):
    words = lemmatize(tokenize(sentence))
    bag = [0] * len(vocabulary)

    for w in words:
        for i, word in enumerate(vocabulary):
            if word == w:
                bag[i] = 1
    return np.array(bag)
