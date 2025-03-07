import ssl
import torch 
import torch.nn as nn    
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# SSL-Fehler umgehen (falls notwendig)
ssl._create_default_https_context = ssl._create_unverified_context

# Device configuration (GPU oder CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameter
input_size = 784  # 28x28 Bilder
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST Dataset laden
train_dataset = torchvision.datasets.MNIST(root="./data", train=True,
    transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root="./data", train=False,
    transform=transforms.ToTensor(), download=True)

# DataLoader für Mini-Batches
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Beispielbilder anzeigen
examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape)  # Erwartete Ausgabe: (100, 1, 28, 28) (100)

# Ein paar Beispielbilder plotten
for i in range(6): 
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap="gray")
plt.show()

# Feedforward Neural Network
class NeuralNet(nn.Module): 
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x): 
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out 
    
# Modell erstellen und auf GPU oder CPU laden
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Verlustfunktion und Optimierer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs): 
    for i, (images, labels) in enumerate(train_loader): 
        # Bilder flach machen (28x28 -> 784) und auf Device schicken
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward Pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")

# Modell testen
with torch.no_grad(): 
    n_correct = 0 
    n_samples = 0 
    for images, labels in test_loader: 
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Vorhersage
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        
        n_samples += labels.size(0)
        n_correct += (predictions == labels).sum().item()
        
    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy = {acc:.2f}%")
