import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
from data_preparation import train_loader, val_loader, class_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained ResNet18 and modify classifier
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # Freeze all layers

# Replace the final layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))  # Output layer matches class count
model = model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Training loop
def train_model(model, epochs=5):
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total
        print(f"[{epoch+1}/{epochs}] Train Loss: {running_loss:.4f}, Accuracy: {train_accuracy:.4f}")

        # Validation
        val_accuracy = evaluate_model(model)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "../models/best_model.pt")
            print("✅ Saved Best Model")

#Model Evaluation
def evaluate_model(model):
    model.eval()
    correct, total = 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))
    return accuracy

# Run training
if __name__ == "__main__":
    train_model(model, epochs=5)
