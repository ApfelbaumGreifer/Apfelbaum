import torch
import torch.nn as nn  
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# CPU als Standard-Device setzen
device = torch.device("cpu")

# MNIST Dataset laden
train_data = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=True, download=True, 
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True
)

test_data = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=False, 
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=False
)

# Modell definieren
class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv_dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 60)
        self.fc2 = nn.Linear(60, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv_dropout(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Modell initialisieren (bleibt auf CPU)
model = Netz()

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8)

# Trainingsfunktion
def train(epoch):
    model.train()
    for batch_id, (data, target) in enumerate(train_data):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, target)
        loss.backward()
        optimizer.step()
        if batch_id % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_id * len(data)}/{len(train_data.dataset)} "
                  f"({100. * batch_id / len(train_data):.0f}%)]\tLoss: {loss.item():.6f}")

# Testfunktion
def test():
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_data:
            out = model(data)
            loss += F.nll_loss(out, target, reduction='sum').item()
            prediction = out.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()
    
    loss /= len(test_data.dataset)
    print(f"Durchschnittlicher Loss: {loss:.4f}")
    print(f"Genauigkeit: {100. * correct / len(test_data.dataset):.2f}%")

# Training und Testing starten
for epoch in range(1, 10):
    train(epoch)
    test()
