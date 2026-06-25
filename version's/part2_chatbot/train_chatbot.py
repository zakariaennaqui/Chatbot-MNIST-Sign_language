import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer

from chatbot_model import build_chatbot

with open("data/intents.json") as f:
    data = json.load(f)

sentences = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences).toarray()

encoder = LabelEncoder()
y = encoder.fit_transform(labels)
y = to_categorical(y)

model = build_chatbot(X.shape[1], y.shape[1])
model.fit(X, y, epochs=200, verbose=0)

model.save("chatbot_model.keras")
np.save("vectorizer.npy", vectorizer.vocabulary_)
np.save("labels.npy", encoder.classes_)
