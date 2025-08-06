import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys
import os

# Define preprocessing – must match training!
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# Class labels (must match training classes order)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # adjust if needed

# Load model architecture
def load_model(model_path):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Run inference on a single image
def predict(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        label = class_names[predicted.item()]
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence.item():.4f}")

# Save TorchScript model
def save_torchscript_model(model, output_path):
    example_input = torch.rand(1, 3, 224, 224)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save(output_path)
    print(f"TorchScript model saved at: {output_path}")

# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    model = load_model("../models/best_model.pt")
    predict(image_path, model)

    # Save TorchScript version (one-time call)
    save_torchscript_model(model, "../models/model_scripted.pt")
