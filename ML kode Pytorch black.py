# Import necessary libraries and scripts
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

# Set standard output and error to handle UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


# Define a simple CNN model in PyTorch
class CNN(nn.Module): 
    def __init__(self): 
        super(CNN, self).__init__() 
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3) 
        self.pool = nn.MaxPool2d(2, 2) 
        self.dropout = nn.Dropout(0.25) 
        self.relu = nn.ReLU()
        
        # Pass a sample input through conv layers to calculate output size
        sample_input = torch.zeros(1, 1, 28, 28)
        conv_output = self._forward_conv(sample_input)
        self.fc1 = nn.Linear(conv_output.numel(), 512)  # Use the dynamic output size
        self.fc2 = nn.Linear(512, 10)

    def _forward_conv(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv3(x)))
        x = self.dropout(x)
        return x
    
    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten for linear layer
        x = self.relu(self.fc1(x)) 
        x = self.fc2(x)
        return x


# Predict the drawn digit and its confidence
def predict_digit(model, image): 
    image = image.resize((28, 28)).convert('L') 
    image = np.array(image).reshape(1, 1, 28, 28) / 255.0 
    image = torch.tensor(image, dtype=torch.float32) 
    image = (image - 0.5) / 0.5  # Normalization for the model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = model.to(device) 
    image = image.to(device) 

    with torch.no_grad(): 
        outputs = model(image) 
    _, predicted = torch.max(outputs.data, 1) 
    confidence = torch.softmax(outputs, dim=1) * 100  # Get confidence as a percentage
    return predicted.item(), confidence[0][predicted].item()  # Return both predicted digit and confidence

# Map digits to their names
digit_to_name = {i: str(i) for i in range(10)} 

# GUI for drawing
class App(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        # Create main frame
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(side=tk.LEFT)

        # Create canvas for drawing
        self.canvas = tk.Canvas(self.main_frame, width=200, height=200, bg='black')  
        self.canvas.pack()

        # Create sidebar frame
        self.sidebar = tk.Frame(self, width=250, bg='lightgrey')
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add labels to sidebar
        self.label_prediction = tk.Label(self.sidebar, text="Prediction: None", font=("Arial", 16), bg='lightgrey')
        self.label_prediction.pack(pady=10)

        self.label_confidence = tk.Label(self.sidebar, text="Confidence: 0%", font=("Arial", 16), bg='lightgrey')
        self.label_confidence.pack(pady=10)

        self.label_correct = tk.Label(self.sidebar, text="Correct Guess: None", font=("Arial", 16), bg='lightgrey')
        self.label_correct.pack(pady=10)

        # Add buttons
        self.button_predict = tk.Button(self.main_frame, text="Predict", command=self.predict)
        self.button_predict.pack(pady=5)

        self.button_clear = tk.Button(self.main_frame, text="Clear", command=self.clear_canvas)
        self.button_clear.pack(pady=5)

        self.button_correct = tk.Button(self.sidebar, text="Correct Guess", command=self.add_to_history)
        self.button_correct.pack(pady=5)

        # History listbox
        self.history_list = tk.Listbox(self.sidebar, height=30, width=50)
        self.history_list.pack(pady=10)

        # Bind paint event
        self.canvas.bind("<B1-Motion>", self.paint)
        
        # Initialize black background for prediction image
        self.image1 = Image.new("L", (200, 200), 0)  
        self.draw = ImageDraw.Draw(self.image1)

    def paint(self, event):
        radius = 10  # Adjust as needed
        x1, y1 = (event.x - radius), (event.y - radius)
        x2, y2 = (event.x + radius), (event.y + radius)
        # Draw on canvas (visible to user)
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', width=0)
        # Draw on the PIL Image (used for prediction)
        self.draw.ellipse([x1, y1, x2, y2], fill='white')

    def clear_canvas(self):
        # Clear both canvas and image used for prediction
        self.canvas.delete("all")
        self.image1 = Image.new("L", (200, 200), 0)
        self.draw = ImageDraw.Draw(self.image1)

        # Reset sidebar
        self.label_prediction.config(text="Prediction: None")
        self.label_confidence.config(text="Confidence: 0%")
        self.label_correct.config(text="Correct Guess: None")

    def predict(self):
        digit, confidence = predict_digit(self.model, self.image1)
        name = digit_to_name[digit]
        self.label_prediction.config(text=f"Prediction: {name}")
        self.label_confidence.config(text=f"Confidence: {confidence:.2f}%")
        # Reset correct guess label
        self.label_correct.config(text="Correct Guess: None")

    def add_to_history(self):
        prediction_text = self.label_prediction.cget("text")
        confidence_text = self.label_confidence.cget("text")
        correct_digit = tk.simpledialog.askstring("Correct Guess", "Enter the correct digit:")
        
        if correct_digit is not None:
            try:
                correct_digit = int(correct_digit)
                if 0 <= correct_digit <= 9:
                    history_entry = f"{prediction_text}, {confidence_text}, Correct: {correct_digit}"
                    self.history_list.insert(tk.END, history_entry)
                    self.label_correct.config(text=f"Correct Guess: {correct_digit}")  # Show the correct guess in the label
                else:
                    tk.messagebox.showerror("Invalid Input", "Please enter a digit between 0 and 9.")
            except ValueError:
                tk.messagebox.showerror("Invalid Input", "Please enter a valid digit.")
        else:
            tk.messagebox.showinfo("Cancelled", "Correct guess entry was cancelled.")

# Transform training data
transform = transforms.Compose([ 
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,)) 
])

# Download and load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform) 
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform) 

# Split training data into training and validation sets
train_size = int(0.8 * len(train_dataset)) 
val_size = len(train_dataset) - train_size 
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size]) 

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model, loss, and optimizer
model = CNN() 
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.000845) 

# Train the model and store loss and accuracy for graphing
def train_model(model, train_loader, val_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = model.to(device) 

    train_losses = [] 
    val_losses = [] 
    train_accuracies = [] 
    val_accuracies = [] 

    for epoch in range(epochs):
        model.train() 
        running_loss = 0.0 
        correct_train = 0  
        total_train = 0 

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) 
            optimizer.zero_grad() 
            outputs = model(images) 
            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step() 
            running_loss += loss.item()

            _, predicted_train = torch.max(outputs.data, 1) 
            total_train += labels.size(0) 
            correct_train += (predicted_train == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        print(f'Epoch {epoch+1}, Training Loss: {train_loss * 100:.2f}%, Training Accuracy: {train_accuracy:.2f}%', flush=True)
        
        # Validation process
        model.eval() 
        val_loss = 0.0 
        correct_val = 0 
        total_val = 0 

        with torch.no_grad():
            for images, labels in val_loader: 
                images, labels = images.to(device), labels.to(device) 
                outputs = model(images) 
                loss = criterion(outputs, labels) 
                val_loss += loss.item() 

                _, predicted_val = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted_val == labels).sum().item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = 100 * correct_val / total_val
        val_accuracies.append(val_accuracy)

        print(f'Validation Loss: {val_loss * 100:.2f}%, Validation Accuracy: {val_accuracy:.2f}%', flush=True)

    # Plot loss and accuracy during training and validation
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.plot(train_losses, label='Training Loss', color='tab:blue')
    ax1.plot(val_losses, label='Validation Loss', color='tab:orange')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)')
    ax2.plot(train_accuracies, label='Training Accuracy', color='tab:green', linestyle='--')
    ax2.plot(val_accuracies, label='Validation Accuracy', color='tab:red', linestyle='--')
    ax2.tick_params(axis='y')

    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.title('Loss and Accuracy During Training and Validation')
    plt.show()
    
# Train the model
train_model(model, train_loader, val_loader)

# Start the application
app = App(model) 
app.mainloop()

torch.save(model.state_dict(), 'digit_model.pth')  # Save the model state