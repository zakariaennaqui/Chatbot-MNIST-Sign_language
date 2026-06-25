import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load saved model
model = tf.keras.models.load_model("mnist_cnn.h5")

# Load and preprocess digit.jpg
img = cv2.imread("digit.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = 255 - img
img = img / 255.0
img = img.reshape(1, 28, 28, 1)

# Show image
plt.imshow(img.reshape(28,28), cmap='gray')
plt.axis('off')
plt.show()

# Predict
prediction = model.predict(img)
print("Recognized digit:", np.argmax(prediction))