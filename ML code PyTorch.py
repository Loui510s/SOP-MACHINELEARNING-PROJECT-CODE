# Importer nødvendige biblioteker og scripts.
import os
import sys
import io
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Ændrer standard output og fejlkode til at kunne behandle UTF-8 indkodning
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Undertrykker andre advarsler for et pænere output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Definerer en simpel CNN model i PyTorch
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.25)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.elu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(self.leaky_relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(self.softmax(self.conv3(x)))
        x = self.dropout(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.softmax(self.fc1(x))
        x = self.fc2(x)
        return x

# Forudsig det tegnede billede
def predict_digit(model, image):
    image = image.resize((28, 28)).convert('L')
    image = np.array(image).reshape(1, 1, 28, 28) / 255.0  # Normalisering
    image = torch.tensor(image, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

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
        digit = predict_digit(self.model, self.image1)
        name = digit_to_name[digit]
        self.title(f"Forudsigelse: {name}")

# Træningsdata transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download og indlæs MNIST datasæt
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Split træningsdata til træning og validering
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialiserer model, loss og optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Træn modellen og gem tab for graf
def train_model(model, train_loader, val_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        print(f'Epoch {epoch+1}, Træningstab: {train_loss}')

        # Validering
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f'Validationstab: {val_loss}, Nøjagtighed: {100 * correct / total}%')

    # Tegn graf for træning og valideringstab
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Træningstab')
    plt.plot(val_losses, label='Validationstab')
    plt.title('Tab under træning og validering')
    plt.xlabel('Epoker')
    plt.ylabel('Tab')
    plt.legend()
    plt.show()

# Træn modellen
train_model(model, train_loader, val_loader)

# Start appen
app = App(model)
app.mainloop()
