import os
import sys
import io
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw

# Reconfigure standard output and error to handle UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Suppress TensorFlow logging for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING logs

# Load the MNIST dataset (handwritten digits)
def load_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    return train_images, train_labels, test_images, test_labels

# Build a simple CNN model
def create_model():
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),  # Add dropout layer for better generalization
        layers.Conv2D(64, (3, 3), activation='leaky_relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), activation='leaky_relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512, activation='exponential'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train the model
def train_model(model, train_images, train_labels, test_images, test_labels):
    history = model.fit(
        train_images, train_labels,
        epochs=10,
        validation_data=(test_images, test_labels),
        verbose=2  # Reduced verbosity to avoid clutter
    )

# Predict drawn image
def predict_digit(model, image):
    image = image.resize((28, 28)).convert('L')  # Convert to grayscale
    image = np.array(image)  # Convert to numpy array
    image = image.reshape(1, 28, 28, 1)  # Reshape for the model
    image = image / 255.0  # Normalize
    prediction = model.predict([image])[0]
    return np.argmax(prediction), max(prediction)

# Map digits to names
digit_to_name = {
    0: "Zero",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine"
}

# GUI for drawing
class App(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.canvas = tk.Canvas(self, width=200, height=200, bg='white')
        self.canvas.pack()
        self.button_predict = tk.Button(self, text="Predict", command=self.predict)
        self.button_predict.pack()
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.image1 = Image.new("RGB", (200, 200), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image1)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=10)
        self.draw.line([x1, y1, x2, y2], fill='black', width=10)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 200, 200], fill="white")

    def predict(self):
        digit, acc = predict_digit(self.model, self.image1)
        name = digit_to_name[digit]  # Map digit to name
        print(f"Predicted Digit: {digit}, Name: {name}, Confidence: {acc:.2f}")  # Debug print
        self.title(f"Prediction: {name}, Confidence: {acc:.2f}")

# Load dataset
train_images, train_labels, test_images, test_labels = load_data()

# Create and train the model
model = create_model()
train_model(model, train_images, train_labels, test_images, test_labels)

# Run the app
app = App(model)
app.mainloop()
