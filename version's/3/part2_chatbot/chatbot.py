import json
import random
import numpy as np
from tensorflow.keras.models import load_model
from nlp_utils import bag_of_words, tokenize, lemmatize

MODEL_PATH = "chatbot_model.h5"

with open("intents.json", encoding="utf-8") as file:
    intents = json.load(file)

words = []
classes = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        words.extend(tokenize(pattern))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = sorted(set(lemmatize(words)))
classes = sorted(set(classes))

model = load_model(MODEL_PATH)

def get_response(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    tag = classes[np.argmax(res)]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

print("🤖 Chatbot municipal prêt (tapez 'quit' pour quitter)")
while True:
    msg = input("Vous : ")
    if msg.lower() == "quit":
        break
    print("Bot :", get_response(msg))
