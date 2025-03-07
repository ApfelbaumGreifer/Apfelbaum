import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter  
import torchvision.utils as vutils

# Setze das Device auf CPU
device = torch.device("cpu")

class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  
        self.fc2 = nn.Linear(256, 6)

    def forward(self, x): 
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  
        x = x.view(x.size(0), -1)  # Flexible Reshape
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  
        x = self.fc2(x)
        return x 

def log_feature_maps(model, images, step, writer):
    """Visualisiert die Feature Maps der ersten Convolution-Schicht in TensorBoard."""
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        x = model.conv1(images)
        x = F.relu(x)
        # Logge die ersten 3 Kanäle von jeweils 8 Bildern
        feature_grid = vutils.make_grid(x[:8, :3], normalize=True, scale_each=True)
        writer.add_image("Feature Maps/Conv1", feature_grid, step)

def train(model, train_loader, optimizer, loss_function, epoch, writer):
    model.train()
    total_loss = 0.0
    for batch_id, (data, target) in enumerate(train_loader): 
        data, target = data.to(device), target.to(device) 
        optimizer.zero_grad()
        out = model(data)
        loss = loss_function(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        global_step = epoch * len(train_loader) + batch_id
        writer.add_scalar("Loss/Train_Batch", loss.item(), global_step)
        if batch_id % 50 == 0:
            log_feature_maps(model, data, global_step, writer)
            print(f"Train Epoch {epoch} [{batch_id * len(data)}/{len(train_loader.dataset)}] "
                  f"Loss: {loss.item():.6f}")
    avg_train_loss = total_loss / len(train_loader)
    writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
    return avg_train_loss

def evaluate(model, loader, loss_function, writer, epoch=None, phase="Validation"):
    """Gemeinsame Funktion zur Validierung und zum Testen."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = loss_function(out, target)
            total_loss += loss.item()
            predictions = out.argmax(dim=1)
            correct += predictions.eq(target).sum().item()
            total += target.size(0)
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    if epoch is not None:
        writer.add_scalar(f"Loss/{phase}", avg_loss, epoch)
        writer.add_scalar(f"Accuracy/{phase}", accuracy, epoch)
        print(f"{phase} Accuracy: {accuracy:.2f}% | {phase} Loss: {avg_loss:.4f}")
    else:
        writer.add_scalar(f"Loss/{phase}", avg_loss)
        writer.add_scalar(f"Accuracy/{phase}", accuracy)
        print(f"{phase} Accuracy: {accuracy:.2f}% | {phase} Loss: {avg_loss:.4f}")
    return accuracy

def main():
    # TensorBoard-Writer initialisieren
    writer = SummaryWriter("runs/schunk_experiment")

    # Pfad zu den Bildern und Transformationen
    data_dir = os.path.join(os.getcwd(), "data", "image")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Bilder laden
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size) 
    test_size = dataset_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # DataLoader
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

    # Modell, Optimizer, Loss und Scheduler definieren
    model = Netz().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Modellstruktur in TensorBoard visualisieren
    dummy_input = torch.randn(1, 3, 128, 128).to(device)
    writer.add_graph(model, dummy_input)

    # Trainingsbilder in TensorBoard speichern (nur zu Beginn)
    dataiter = iter(train_loader)
    images, _ = next(dataiter)
    img_grid = vutils.make_grid(images[:8])  
    writer.add_image("Training Images", img_grid)

    # Trainingseinstellungen
    num_epochs = 100
    best_val_acc = 0.0
    early_stop_patience = 20
    epochs_without_improve = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, loss_function, epoch, writer)
        val_acc = evaluate(model, val_loader, loss_function, writer, epoch, phase="Validation")

        scheduler.step()

        # Early Stopping: Training abbrechen, wenn keine Verbesserung über mehrere Epochen erfolgt
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Bestes Modell gespeichert!")
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= early_stop_patience:
                print("Keine Verbesserung über mehrere Epochen. Training wird frühzeitig beendet.")
                break

    # Bestes Modell laden und testen
    model.load_state_dict(torch.load("best_model.pth"))
    evaluate(model, test_loader, loss_function, writer, phase="Test")
    writer.close()

if __name__ == '__main__':
    main()
