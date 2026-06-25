import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("../part1_mnist/saved_model/mnist_cnn.keras")

# Load MNIST test data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1)  # add channel dimension

# Make predictions
predictions = model.predict(x_test)

# Function to display images with predictions
def plot_predictions(images, labels, preds, num=10):
    plt.figure(figsize=(15, 4))
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"Pred: {np.argmax(preds[i])}\nTrue: {labels[i]}")
        plt.axis('off')
    plt.show()

# Show first 10 test images with predictions
plot_predictions(x_test, y_test, predictions, num=20)
