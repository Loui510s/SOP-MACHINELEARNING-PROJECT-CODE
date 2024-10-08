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
class CNN(nn.Module): #Opretter en CNN klasse
    def __init__(self): #Definerer __init__ funktionen
        super(CNN, self).__init__() #laver en super
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) # Laver det første konvolutionerende lag
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) # Laver det andet konvolutionerende lag
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3) # Laver det tredje konvolutionerende lag
        self.pool = nn.MaxPool2d(2, 2) # Nedskalere dataens kanaler
        self.fc1 = nn.Linear(128, 512) # Laver en lineær funktion
        self.fc2 = nn.Linear(512, 10) # Laver en lineær funktion Igen
        self.dropout = nn.Dropout(0.25) #smider en lille smule af kanalerne ud
        self.relu = nn.ReLU() #Laver en relu funktion
        self.leaky_relu = nn.LeakyReLU(0.2) # Laver en Leaky_ReLU funktion
        self.elu = nn.ELU() #Laver en ELU funktion
        self.softmax = nn.Softmax(dim=1) #Laver en softmax funktion
        self.mish = nn.Mish()  #laver en Mish funktion

    def forward(self, x): #Definerer forward funktionen, her findes rækkefølgen som __init__ funktionen følger
        x = self.pool(self.mish(self.conv1(x))) # Aktivere Mish funktionen
        x = self.dropout(x) # Aktivere dropoutlaget
        x = self.pool(self.leaky_relu(self.conv2(x))) # Aktivere Leaky_ReLU funktionen
        x = self.dropout(x) # Aktivere dropout laget igen
        x = self.pool(self.softmax(self.conv3(x))) # Aktiverer Softmax funktionen
        x = self.dropout(x) # Aktiverer dropout funktionen

        x = x.view(x.size(0), -1)  # Flader størrelsen af x ud
        x = self.softmax(self.fc1(x)) # Aktiverer Softmax funktionen
        x = self.fc2(x) # aktiverer den anden Lineær funktion
        return x # Tilbagekalder x

# Forudsig det tegnede billede
def predict_digit(model, image): # Definere forudsignings funktionen.
    image = image.resize((28, 28)).convert('L') # Nedskalerer billederne
    image = np.array(image).reshape(1, 1, 28, 28) / 255.0  # Normalisere billederne
    image = torch.tensor(image, dtype=torch.float32) # Sørger for at der ikke sker kopieringer af billeder.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Definere hvilken del af computeren der skal bruges på at kører programmet.
    model = model.to(device) # Definerer at modellen skal fittes til devicet
    image = image.to(device) # Definerer at billedet skal fittes til devicet

    with torch.no_grad(): # Stopper gradient beregning
        outputs = model(image) # Definerer output
    _, predicted = torch.max(outputs.data, 1) # Definerer forudsigelsen
    return predicted.item() # Tilbagekalder predicted.item()

# Laver tal om til navne
digit_to_name = {i: str(i) for i in range(10)} # omnavngiver tallende til navne

# GUI for tegning
class App(tk.Tk): # opretter en klasse til applikationen
    def __init__(self, model): # Definerer __init__ funktionen
        super().__init__() #Definere superen for __init__ funktionen
        self.model = model # Definere modellen
        self.canvas = tk.Canvas(self, width=200, height=200, bg='white') # Laver et canvas
        self.canvas.pack() # Laver en canvas pakke
        self.button_predict = tk.Button(self, text="Predict", command=self.predict) # Laver en knap til at forudsige tegningen
        self.button_predict.pack() # Laver en forudsigelses pakke
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_canvas) # Laver en knap til at rydde tegningen
        self.button_clear.pack() # Laver en knap til at rydde pakke

        self.canvas.bind("<B1-Motion>", self.paint) # Laver en binding af tegningen til 
        self.image1 = Image.new("RGB", (200, 200), (255, 255, 255)) # Definere farven der skal bruges til at farve tegningen over med
        self.draw = ImageDraw.Draw(self.image1) # laver en tegne funktion

    def paint(self, event): # definerer malingen
        x1, y1 = (event.x - 1), (event.y - 1) # fortæller hvor koordinaterne er 
        x2, y2 = (event.x + 1), (event.y + 1) # fortæller hvor koordinaterne er 
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=10) # Definerer størrelsen af malingen
        self.draw.line([x1, y1, x2, y2], fill='black', width=10) # Definerer farven af tegningen og tykkelsen af linjerne

    def clear_canvas(self): # definerer rydnings funktionen af canvasset
        self.canvas.delete("all") # Laver en funktion der sletter alt
        self.draw.rectangle([0, 0, 200, 200], fill="white") # tegner en hvid rektangel på størrelse med canvasset

    def predict(self): # Definerer forudsigelses funktionen
        digit = predict_digit(self.model, self.image1) # Definerer tal
        name = digit_to_name[digit] # Definerer navn
        self.title(f"Forudsigelse: {name}") # Laver en titel til forudsigelsen

# Træningsdata transformering
transform = transforms.Compose([ #laver en compositions transformering
    transforms.ToTensor(), # Konverterer et PIL billede til en FloatTensor
    transforms.Normalize((0.5,), (0.5,)) # Nomaliserer billedets data
])

# Download og indlæs MNIST datasæt
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform) #Downloader Træningsdatasættet
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform) #Downloader test datasættet

# Split træningsdata til træning og validering
train_size = int(0.8 * len(train_dataset)) #Fortæller hvor meget af trænings sættet der skal bruges
val_size = len(train_dataset) - train_size # Fortæller størrelsen af trænings sættet der skal bruges til validerings sættet
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size]) # splitter trænings- og valideringssættet

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True) #Loader træningssættet
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False) #Loader valideringssættet
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False) #Loader testsættet

# Initialiserer model, loss og optimizer
model = CNN() # Fortæller at CNN er en model
criterion = nn.CrossEntropyLoss() # Opstiller en funktion for kriterier af entropi tab
optimizer = optim.Adam(model.parameters(), lr=0.00085) # Laver en optimerings funktion

# Træn modellen og gem tab og nøjagtighed for graf
def train_model(model, train_loader, val_loader, epochs=10): # definerer træningsmodellen
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # fortæller om den skal bruge GPU eller CPU
    model = model.to(device) # Fortæller hvad modellen er

    train_losses = [] #laver en træningstabs variabel
    val_losses = [] # laver en validerings tabs variabel
    train_accuracies = [] #laver en trænings præcisions variabel
    val_accuracies = [] #laver en validerings præcisions variabel

    for epoch in range(epochs): #definerer epoker
        model.train() #indlæser træningsmodellen modellen
        running_loss = 0.0 #fortæller start tabene
        correct_train = 0  #fortæller starten med korrekt trænet
        total_train = 0 #fortæller starten med totalt trænet

        for images, labels in train_loader: #laver et for loop med billeder og labels i træningsloaderen
            images, labels = images.to(device), labels.to(device) # fortæller hvilke labels og images der skal bruges
            
            optimizer.zero_grad() #definerer optimizeren
            outputs = model(images)  #Definerer outputs
            loss = criterion(outputs, labels) #Definerer tab
            loss.backward() #indlæser tidligere tab
            optimizer.step() #optimisere
            running_loss += loss.item() #fortæller hvor stor en loss der er

            _, predicted_train = torch.max(outputs.data, 1) # laver en forudsigelse ud af det trænede data
            total_train += labels.size(0) # laver labels til de trænede modeller
            correct_train += (predicted_train == labels).sum().item() # vurderer hvilke forudsigelser der er korrekte ift. labelsne
         
        train_loss = running_loss / len(train_loader) # laver en trainingloss funktion
        train_losses.append(train_loss)s # læser tabne på træningssættet
        train_accuracy = 100 * correct_train / total_train #laver en procentvis udregning af præcisionen for testne
        train_accuracies.append(train_accuracy) # aflæs det samlede præcision af træningerne

        print(f'Epoch {epoch+1}, Træningstab: {train_loss}, Træningsnøjagtighed: {train_accuracy}%') #printer epokerne med deres præcision og tab
    
        # Validering
        model.eval() #kører evaluerings modellen
        val_loss = 0.0 #fortæller start validerings tab
        correct_val = 0 #Fortæller start korrekte valideringer
        total_val = 0 #Fortæller den totale validerings antal

        with torch.no_grad(): #Fortæller hvilken torch funktion der skal bruges
            for images, labels in val_loader: # fortæller hvad der skal bruges fra validerings loaderen
                images, labels = images.to(device), labels.to(device) # fortæller hvor billederne og labelsne skal ligge.
                outputs = model(images) # definerer output
                loss = criterion(outputs, labels) #definerer tabskriterierne
                val_loss += loss.item() # Definerer validerings tab

                _, predicted_val = torch.max(outputs.data, 1) #fortæller hvad forudsigelsen af valideringen er baseret på
                total_val += labels.size(0) # fortæller hvad det totale antal af valideringen er.
                correct_val += (predicted_val == labels).sum().item() # fortæller hvor mange korrekte valideringer der er.

        val_loss /= len(val_loader) # fortæller hvor stort et tab der er i valideringen
        val_losses.append(val_loss) # fortæller det samlede antal tab i valideringen
        val_accuracy = 100 * correct_val / total_val # fortæller præcisionen i procent
        val_accuracies.append(val_accuracy) # fortæller den samlede antal præcision

        print(f'Validationstab: {val_loss}, Valideringsnøjagtighed: {val_accuracy}%') #printer epokerne af valideringen med dets tab og præcision

    # Tegn graf for tab og nøjagtighed under træning og validering
    fig, ax1 = plt.subplots(figsize=(10, 5))  #laver en figur til at plotte grafer

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

# Start appen
app = App(model) # Fortæller hvad appen er
app.mainloop() #kører appen
