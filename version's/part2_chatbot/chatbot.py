import json
import random
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

model = tf.keras.models.load_model("chatbot_model.keras")

with open("data/intents.json") as f:
    data = json.load(f)

vectorizer = CountVectorizer(vocabulary=np.load("vectorizer.npy", allow_pickle=True).item())
encoder = LabelEncoder()
encoder.classes_ = np.load("labels.npy", allow_pickle=True)

def chatbot_response(text):
    X = vectorizer.transform([text]).toarray()
    prediction = model.predict(X)
    tag = encoder.inverse_transform([prediction.argmax()])[0]

    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

while True:
    msg = input("Vous: ")
    if msg.lower() == "quit":
        break
    print("Bot:", chatbot_response(msg))
