import json
import random
import numpy as np
from tensorflow.keras.optimizers import Adam
from model import build_model
from nlp_utils import tokenize, lemmatize

MODEL_PATH = "chatbot_model.h5"

with open("intents.json", encoding="utf-8") as file:
    intents = json.load(file)

words = []
classes = []
documents = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = sorted(set(lemmatize(words)))
classes = sorted(set(classes))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = lemmatize(doc[0])
    for w in words:
        bag.append(1 if w in word_patterns else 0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

model = build_model(len(train_x[0]), len(train_y[0]))

model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_x, train_y, epochs=200, batch_size=5)
model.save(MODEL_PATH)

print("✅ Modèle chatbot entraîné et sauvegardé")
