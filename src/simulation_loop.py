import os
import csv
import torch # type: ignore
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore

# TorchScript model path
MODEL_PATH = "../models/model_scripted.pt"

# Input folder with images simulating the conveyor
INPUT_FOLDER = "../data/conveyor_simulation/"  # Change if needed
OUTPUT_CSV = "../results/predictions.csv"

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.6

# Class labels (must match training)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # adjust if needed

# Preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# Load scripted model
model = torch.jit.load(MODEL_PATH)
model.eval()

# Create results folder if missing
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Prepare CSV file
with open(OUTPUT_CSV, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image', 'Prediction', 'Confidence'])

    # Simulate frame-by-frame processing
    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(INPUT_FOLDER, filename)
            image = Image.open(image_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

                label = class_names[predicted.item()]
                conf_value = confidence.item()

                # Print result
                print(f" {filename} →  {label} ({conf_value:.4f})", end=' ')
                if conf_value < CONFIDENCE_THRESHOLD:
                    print("⚠️ LOW CONFIDENCE")
                else:
                    print()

                # Write to CSV
                writer.writerow([filename, label, f"{conf_value:.4f}"])
