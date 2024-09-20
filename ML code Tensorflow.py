# Importere nødvændige biblioteker og scripts.
import os
import sys
import io
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Ændre standard output og fejlkode til at kunne behandle UTF-8 indkodning
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Undertrykker TensorFlow logning for et pænere output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO and WARNING logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disables oneDNN optimizations

# Indlæser MNIST datasættet
def load_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    return train_images, train_labels, test_images, test_labels

# Bygger en simpel CNN model
def create_model():
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation='exponential'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), activation='leaky_relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, loss_scale_factor=1),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Træn modellen og gem træningshistorikken
def train_model(model, train_images, train_labels, val_images, val_labels):
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels), verbose=2)
    return history

# Plot læringskurver (træning og validering for både nøjagtighed og tab)
def plot_learning_curves(history):
    # Plot trænings- og valideringsnøjagtighed
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Træningsnøjagtighed')
    plt.plot(history.history['val_accuracy'], label='Valideringsnøjagtighed')
    plt.title('Træning og Validering Nøjagtighed')
    plt.xlabel('Epoker')
    plt.ylabel('Nøjagtighed')
    plt.legend()

    # Plot trænings- og valideringstab
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Træningstab')
    plt.plot(history.history['val_loss'], label='Valideringstab')
    plt.title('Træning og Validering Tab')
    plt.xlabel('Epoker')
    plt.ylabel('Tab')
    plt.legend()

    # Vise begge plots
    plt.tight_layout()
    plt.show()

# Forudsig det tegnede billede
def predict_digit(model, image):
    image = image.resize((28, 28)).convert('L')  # Convert to grayscale
    image = np.array(image).reshape(1, 28, 28, 1) / 255.0  # Normalize
    prediction = model.predict(image)[0]
    return np.argmax(prediction), max(prediction)

# Laver tal om til navne
digit_to_name = {i: str(i) for i in range(10)}

# GUI for tegning
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
        name = digit_to_name[digit]
        self.title(f"Prediction: {name}, Confidence: {acc:.2f}")

# Indlæs datasæt
train_images, train_labels, test_images, test_labels = load_data()

# Split træningsdata i trænings- og valideringssæt (80% træning, 20% validering)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Opret og træn modellen
model = create_model()
history = train_model(model, train_images, train_labels, val_images, val_labels)

# Plot læringskurver
plot_learning_curves(history)

# Kør appen
app = App(model)
app.mainloop()
