import torch
from torchvision import datasets, transforms
import os

# Daten-Transformationen für Training & Validierung
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Verzeichnis mit den Bildern
data_dir = 'hymenoptera_data'

# Lade die Bilder in PyTorch Datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

# Lade die Bilder in DataLoader
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=0)
               for x in ['train', 'val']}


# Anzahl der Bilder in jedem Set
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(f"Trainingsbilder: {dataset_sizes['train']}, Validierungsbilder: {dataset_sizes['val']}")
print(f"Klassen: {class_names}")
