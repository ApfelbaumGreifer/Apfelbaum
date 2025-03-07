import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import time
import os
import ssl
from torch.utils.tensorboard import SummaryWriter

# TensorBoard Writer
writer = SummaryWriter(log_dir="logs")

# SSL-Probleme umgehen
ssl._create_default_https_context = ssl._create_unverified_context

from dataset_loader import dataloaders, dataset_sizes, class_names  # Lade vorherige Variablen

# Prüfe, ob CUDA verfügbar ist, ansonsten CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Lade das vortrainierte ResNet18-Modell
model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features

# Ersetze die letzte Schicht mit einer für 2 Klassen (Ameisen, Bienen)
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

# Verlustfunktion & Optimizer definieren
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Lernraten-Scheduler (reduziert LR alle 7 Epochen um den Faktor 0.1)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 🔹 **Funktion train_model zuerst definieren**
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 🔥 TensorBoard Logging
            writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
            writer.add_scalar(f"Accuracy/{phase}", epoch_acc, epoch)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f}')  # ✅ HIER WAR DER FEHLER

    model.load_state_dict(best_model_wts)

    # 🔥 TensorBoard schließen
    writer.close()

    return model

# 🔹 **Jetzt erst das Training starten**
if __name__ == "__main__":
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)
