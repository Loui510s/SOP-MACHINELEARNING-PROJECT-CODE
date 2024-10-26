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
        self.relu = nn.ReLU() 
        self.leaky_relu = nn.LeakyReLU(0.2) 
        self.elu = nn.ELU() 
        self.softmax = nn.Softmax(dim=1) 
        self.mish = nn.Mish() 

    #Definerer forward funktionen, her findes rækkefølgen som __init__ funktionen følger
    def forward(self, x): 
        x = self.pool(self.mish(self.conv1(x))) 
        x = self.dropout(x) 
        x = self.pool(self.leaky_relu(self.conv2(x))) 
        x = self.dropout(x) 
        x = self.pool(self.softmax(self.conv3(x))) 
        x = self.dropout(x) 

        x = x.view(x.size(0), -1)  
        x = self.softmax(self.fc1(x)) 
        x = self.fc2(x) 
        return x

# Forudsig det tegnede billede
def predict_digit(model, image): 
    image = image.resize((28, 28)).convert('L') 
    image = np.array(image).reshape(1, 1, 28, 28) / 255.0 
    image = torch.tensor(image, dtype=torch.float32) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = model.to(device) 
    image = image.to(device) 

    with torch.no_grad(): 
        outputs = model(image) 
    _, predicted = torch.max(outputs.data, 1) 
    return predicted.item() 

# angiver tallene som navne
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

    def paint(self, event): # definerer malingen
        x1, y1 = (event.x - 1), (event.y - 1) 
        x2, y2 = (event.x + 1), (event.y + 1) 
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=10) 
        self.draw.line([x1, y1, x2, y2], fill='black', width=10) 

    def clear_canvas(self): # definerer rydnings funktionen af canvasset
        self.canvas.delete("all") 
        self.draw.rectangle([0, 0, 200, 200], fill="white") 

    def predict(self): # Definerer forudsigelses funktionen
        digit = predict_digit(self.model, self.image1) 
        name = digit_to_name[digit] 
        self.title(f"Forudsigelse: {name}") 

# Transformerer Træningsdataen
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

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True) #Loader træningssættet
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False) #Loader valideringssættet
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False) #Loader testsættet

# Initialiserer model, loss og optimizer
model = CNN() 
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.00085) 

# Træn modellen og gem tab og nøjagtighed for graf
def train_model(model, train_loader, val_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = model.to(device) 

    train_losses = [] 
    val_losses = [] 
    train_accuracies = [] 
    val_accuracies = [] 

    for epoch in range(epochs): #definerer epoker
        model.train() 
        running_loss = 0.0 
        correct_train = 0  
        total_train = 0 

        for images, labels in train_loader: #laver et for loop med billeder og labels i træningsloaderen
            images, labels = images.to(device), labels.to(device) 
            
            optimizer.zero_grad() 
            outputs = model(images)  
            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step() 
            running_loss += loss.item()

            _, predicted_train = torch.max(outputs.data, 1) # laver en forudsigelse ud af det trænede data
            total_train += labels.size(0) # laver labels til de trænede modeller
            correct_train += (predicted_train == labels).sum().item() # vurderer hvilke forudsigelser der er korrekte ift. labelsne
         
        train_loss = running_loss / len(train_loader) # laver en trainingloss funktion
        train_losses.append(train_loss)s # læser tabne på træningssættet
        train_accuracy = 100 * correct_train / total_train #laver en procentvis udregning af præcisionen for testne
        train_accuracies.append(train_accuracy) # aflæs det samlede præcision af træningerne

        print(f'Epoch {epoch+1}, Træningstab: {train_loss}, Træningsnøjagtighed: {train_accuracy}%') #udskriver epokerne med deres præcision og tab i terminalen
    
        # Kører en valideringsprocess
        model.eval() 
        val_loss = 0.0 
        correct_val = 0 
        total_val = 0 

        with torch.no_grad(): #Fortæller hvilken torch funktion der skal bruges
            for images, labels in val_loader: 
                images, labels = images.to(device), labels.to(device) 
                outputs = model(images) 
                loss = criterion(outputs, labels) 
                val_loss += loss.item() 

                _, predicted_val = torch.max(outputs.data, 1) #fortæller hvad forudsigelsen af valideringen er baseret på
                total_val += labels.size(0) # fortæller hvad det totale antal af valideringen er.
                correct_val += (predicted_val == labels).sum().item() # fortæller hvor mange korrekte valideringer der er.

        val_loss /= len(val_loader) # fortæller hvor stort et tab der er i valideringen
        val_losses.append(val_loss) # fortæller det samlede antal tab i valideringen
        val_accuracy = 100 * correct_val / total_val # fortæller præcisionen i procent
        val_accuracies.append(val_accuracy) # fortæller den samlede antal præcision

        print(f'Validationstab: {val_loss}, Valideringsnøjagtighed: {val_accuracy}%') #udskriver epokerne af valideringen med dets tab og præcision i terminalen

    # Tegn graf for tab og nøjagtighed under træning og validering
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Epoker') #laver x-aksens titel til epoker
    ax1.set_ylabel('Tab') #laver y-aksens titel til tab
    ax1.plot(train_losses, label='Træningstab', color='tab:blue') # plotter træningstabne
    ax1.plot(val_losses, label='Validationstab', color='tab:orange') # plotter valideringstabne
    ax1.tick_params(axis='y') # fortæller hvor høj y-aksen skal være

    ax2 = ax1.twinx() #duplikerer den første grafs base under nyt navn
    ax2.set_ylabel('Nøjagtighed (%)') #laver y-aksens titel om til præcision
    ax2.plot(train_accuracies, label='Træningsnøjagtighed', color='tab:green', linestyle='--') #plotter træningspræcisionen
    ax2.plot(val_accuracies, label='Valideringsnøjagtighed', color='tab:red', linestyle='--') #plotter valideringspræcisionen
    ax2.tick_params(axis='y') #fortæller hvor høj y-aksen skal være

    fig.tight_layout() #fortæller at graferne skal ligge tæt

    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9)) # placerer titlen på graferne
    plt.title('Tab og nøjagtighed under træning og validering') # laver navn på titlen
    plt.show() # viser grafen

# Træn modellen
train_model(model, train_loader, val_loader) # Træner modellen

# Start applicationen
app = App(model) 
app.mainloop() 
