import nltk
from nltk.stem import WordNetLemmatizer
import json

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def load_intents(path):
    with open(path) as f:
        return json.load(f)

def preprocess(sentence):
    tokens = nltk.word_tokenize(sentence.lower())
    return [lemmatizer.lemmatize(w) for w in tokens]
