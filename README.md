# ?? Chatbot + MNIST + Sign Language Recognition

> A Python AI course project combining **NLP-powered Chatbot**, **MNIST digit recognition** and **Sign Language digit classification** — all built from scratch using PyTorch and TensorFlow/Keras.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?logo=pytorch)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![MySQL](https://img.shields.io/badge/MySQL-Database-blue?logo=mysql)](https://www.mysql.com/)

---

## ?? Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Module 1 — NLP Chatbot](#module-1--nlp-chatbot)
- [Module 2 — MNIST Digit Recognition](#module-2--mnist-digit-recognition-cnn)
- [Module 3 — Sign Language Classification](#module-3--sign-language-digit-classification)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)

---

## Overview

This project was developed as part of the **"Programmation Python Pour IA"** course (S7 — ENSA Berrechid).
It is split into **three independent AI modules**, each exploring a different aspect of applied machine learning:

| Module | Task | Framework |
|--------|------|-----------|
| ??? Chatbot | Intent classification via NLP + MLP | PyTorch + NLTK + MySQL |
| ?? MNIST | Handwritten digit recognition | TensorFlow / Keras CNN |
| ?? Sign Language | Sign language digit classification (0–9) | TensorFlow / Keras CNN |

---

## Project Structure

```
project/
+-- main.py                   # Base chatbot prototype (standalone, no DB)
+-- intents.json              # Base intents (greeting, goodbye, stocks...)
+-- mnist_cnn.py              # Quick MNIST demo script
+-- MNIST.pdf                 # MNIST reference report
¦
+-- chatbot/                  # ??? Full chatbot module (GUI + DB)
¦   +-- main.py               # Tkinter GUI — login + chatbot window
¦   +-- chatbot.py            # ChatbotModel (PyTorch MLP) + ChatbotAssistant
¦   +-- database.py           # MySQL: user auth, chat history, dynamic training data
¦   +-- intents.json          # Minimal intents (greeting, goodbye)
¦   +-- chatbot_model.pth     # Saved PyTorch model weights
¦   +-- chatbot.sql           # Full MySQL schema
¦
+-- mnist/                    # ?? MNIST CNN module
¦   +-- mnist_cnn.py          # CNN model + training + custom image test (OpenCV)
¦   +-- model.py              # Model definition
¦   +-- train.py              # Training script
¦   +-- evaluate.py           # Evaluation script
¦   +-- predict.py            # Single-image prediction
¦   +-- visualize_mnist.py    # Visualization of predictions
¦   +-- mnist_cnn.h5          # Saved Keras model
¦   +-- digit.jpg / digit.png # Real-world test images
¦
+-- sign_language/            # ?? Sign Language CNN module
    +-- model.py              # CNN architecture (3× Conv2D, 64×64 RGB, 10 classes)
    +-- train_sign_digits.py  # Training script
    +-- evaluate_sign_digits.py
    +-- visualize_sign_digits.py  # 6×6 prediction grid
    +-- utils.py              # Data loading utilities
    +-- data/                 # Train/test image folders (0–9 digit signs)
    +-- saved_model/          # Saved sign_digits_cnn.keras
```

---

## Module 1 — NLP Chatbot

### Description

A fully functional **intent-based chatbot** with a Tkinter desktop GUI, featuring:
- **NLP pipeline**: normalisation ? tokenisation ? lemmatisation (NLTK) ? Bag-of-Words vectorisation
- A **PyTorch MLP** trained on intent patterns from `intents.json` and DB-extracted keywords
- **MySQL database** for user authentication, chat history, and **dynamic training data** (events, municipal services, transports)
- **Hybrid intent detection**: direct DB keyword matching + ML model fallback

### Architecture

```
User Input
    ?
normalize_text() ? NLTK tokenize + lemmatize ? Bag of Words vector
    ?
detect_table_intent() --- direct DB keyword match ---? DB query ? formatted response
    ? (fallback)
ChatbotModel (PyTorch MLP)
    fc1(input ? 128) ? ReLU ? Dropout(0.5)
    fc2(128 ? 64)    ? ReLU ? Dropout(0.5)
    fc3(64 ? n_intents)
    ?
argmax ? predicted intent ? response
```

### Database Schema

```sql
users               -- id, username, email, password
messages            -- id, user_id, message, response, timestamp
evenements          -- id, nom, description, lieu, date
services_municipaux -- id, nom, horaires, adresse, telephone, description
transports          -- id, ligne, type, destination, horaires, tarif, etat_trafic
```

---

## Module 2 — MNIST Digit Recognition (CNN)

Trains a CNN on the classic [MNIST](http://yann.lecun.com/exdb/mnist/) dataset (60k handwritten digits, 28×28 grayscale).

### CNN Architecture

```
Input (28×28×1)
  ? Conv2D(32, 3×3, ReLU) ? MaxPooling(2×2)
  ? Conv2D(64, 3×3, ReLU) ? MaxPooling(2×2)
  ? Flatten ? Dense(128, ReLU) ? Dense(10, Softmax)
```

- **Optimizer**: Adam | **Loss**: Sparse Categorical Cross-Entropy | **Epochs**: 5
- Real-world inference on custom `digit.jpg` via OpenCV preprocessing
- Saves model as `mnist_cnn.h5`

---

## Module 3 — Sign Language Digit Classification

Trains a CNN to classify **sign language hand gesture images** (digits 0–9) from the [Sign Language Digits Dataset](https://github.com/ardamavi/Sign-Language-Digits-Dataset).

### CNN Architecture

```
Input (64×64×3 RGB)
  ? Conv2D(32, 3×3, ReLU)  ? MaxPooling(2×2)
  ? Conv2D(64, 3×3, ReLU)  ? MaxPooling(2×2)
  ? Conv2D(128, 3×3, ReLU) ? MaxPooling(2×2)
  ? Flatten ? Dense(128, ReLU) ? Dense(10, Softmax)
```

- **Optimizer**: Adam | **Loss**: Sparse Categorical Cross-Entropy | **Epochs**: 10
- Images loaded via `image_dataset_from_directory` (directory-structured: `data/train/0..9`, `data/test/0..9`)
- `visualize_sign_digits.py` renders a 6×6 prediction grid (green = correct, red = wrong)
- Saves model as `saved_model/sign_digits_cnn.keras`

---

## Tech Stack

| Technology | Use |
|------------|-----|
| **Python 3.9+** | Core language |
| **PyTorch** | Chatbot MLP (training + inference) |
| **TensorFlow / Keras** | CNN models (MNIST & Sign Language) |
| **NLTK** | Tokenization, lemmatization |
| **MySQL (mysql-connector-python)** | Chatbot database |
| **Tkinter** | Chatbot desktop GUI |
| **OpenCV (cv2)** | Custom digit image preprocessing |
| **NumPy / Matplotlib** | Arrays & visualization |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/zakariaennaqui/Chatbot-MNIST-Sign_language.git
cd Chatbot-MNIST-Sign_language
```

### 2. Install dependencies

```bash
pip install torch torchvision nltk tensorflow opencv-python mysql-connector-python matplotlib numpy
```

### 3. Download NLTK data

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

### 4. Set up MySQL (chatbot module only)

```bash
mysql -u root -p < chatbot/chatbot.sql
```

> Update credentials in `chatbot/database.py` if needed.

---

## Usage

### Run the Chatbot GUI

```bash
cd chatbot
python main.py
```

Log in or register ? ask about events, services or transport lines.

### Train & Test MNIST

```bash
cd mnist
python mnist_cnn.py
```

### Train & Visualize Sign Language

```bash
cd sign_language
python train_sign_digits.py
python visualize_sign_digits.py
```

---

## Architecture Overview

```
+--------------------------------------------------------------+
¦                     Python AI Project                        ¦
¦                                                              ¦
¦  +-------------------+  +------------+  +----------------+  ¦
¦  ¦  ??? NLP Chatbot    ¦  ¦ ?? MNIST   ¦  ¦ ?? Sign Language¦  ¦
¦  ¦                   ¦  ¦            ¦  ¦                ¦  ¦
¦  ¦  PyTorch MLP      ¦  ¦  Keras CNN ¦  ¦  Keras CNN     ¦  ¦
¦  ¦  NLTK + BoW       ¦  ¦  28×28 BW  ¦  ¦  64×64 RGB     ¦  ¦
¦  ¦  MySQL + Tkinter  ¦  ¦  10 digits ¦  ¦  10 gestures   ¦  ¦
¦  +-------------------+  +------------+  +----------------+  ¦
+--------------------------------------------------------------+
```

---

## Author

**Zakaria Ennaqui**
4th-year Computer Engineering Student — ENSA Berrechid
?? [github.com/zakariaennaqui](https://github.com/zakariaennaqui) | ?? [zakaria-ennaqui.vercel.app](https://zakaria-ennaqui.vercel.app)
