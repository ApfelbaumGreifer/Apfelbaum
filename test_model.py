import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from train_model import model_ft, class_names

# Stelle sicher, dass das Modell im Evaluierungsmodus ist
model_ft.eval()

# Bildtransformationen (müssen die gleichen wie beim Training sein)
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Bildpfad (ersetze dies mit einem echten Bild aus deinem Dataset oder einem eigenen Bild!)
image_path = "hymenoptera_data/val/bees/72100438_73de9f17af.jpg"

# Bild laden und transformieren
image = Image.open(image_path)
image_tensor = data_transforms(image).unsqueeze(0)

# Vorhersage machen
with torch.no_grad():
    outputs = model_ft(image_tensor)
    _, preds = torch.max(outputs, 1)

# Ergebnis anzeigen
plt.imshow(image)
plt.title(f"Predicted: {class_names[preds[0]]}")
plt.show()
