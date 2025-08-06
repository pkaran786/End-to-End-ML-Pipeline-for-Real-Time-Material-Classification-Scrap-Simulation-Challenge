import os
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from collections import Counter

data_dir = "../data"
def clean_dataset(data_dir):
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Will throw error if corrupt
            except (IOError, SyntaxError):
                print(f"Removing corrupted file: {img_path}")
                os.remove(img_path)

clean_dataset(data_dir)

# Standard preprocessing transform
preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize([0.5, 0.5, 0.5],  # Normalize (mean, std) per channel
                         [0.5, 0.5, 0.5])
])

augment_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

train_dataset = ImageFolder(root="../data", transform=augment_transform)


class_names = train_dataset.classes
print("Classes:", class_names)
print("Number of samples:", len(train_dataset))

dataset = ImageFolder(root="../data", transform=augment_transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# for images, labels in train_loader:
#     print(f"Image batch shape: {images.shape}")
#     print(f"Labels: {labels}")
#     break
