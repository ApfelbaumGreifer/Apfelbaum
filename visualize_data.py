import matplotlib.pyplot as plt
import numpy as np
import torchvision

from dataset_loader import dataloaders, class_names  # Vorherigen Code wiederverwenden

# Funktion zum Anzeigen von Bildern
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # Damit das Bild aktualisiert wird

if __name__ == '__main__':
    # Ein Batch von Trainingsbildern abrufen
    inputs, classes = next(iter(dataloaders['train']))

    # Einen Grid aus Bildern erstellen
    out = torchvision.utils.make_grid(inputs)

    # Anzeigen
    imshow(out, title=[class_names[x] for x in classes])
    plt.show()
