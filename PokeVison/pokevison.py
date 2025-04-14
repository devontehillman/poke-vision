'''
Authors: Evan Gronewold, Devonte Hillman, Svens Daukss
Overview:
    This script implements a Pokémon image classification pipeline using PyTorch. It includes functionality for 
    data preprocessing, dataset splitting, model training, evaluation, and prediction. The model is based on 
    a ResNet-50 architecture, fine-tuned for the classification of Pokémon images into predefined classes.
Modules: #XX Might Delete
    os: For directory and file operations.
    shutil: For file and directory manipulation.
    PIL: For image processing.
    torchvision: For dataset and model utilities.
    torch: For deep learning operations.
    sklearn.metrics: For evaluation metrics like confusion matrix and classification report.
    numpy: For numerical operations.
Classes:
    PokemonClassifier: A custom PyTorch model class that fine-tunes ResNet-50 for Pokémon classification.
Functions:
    create_directory_structure(): Creates the directory structure for train, validation, and test datasets.
    split_dataset(): Splits a raw dataset into train, validation, and test sets with specified ratios.
    train_model(): Trains the Pokémon classifier model and saves the best-performing model.
    predict_pokemon(): Predicts the class of a given Pokémon image and returns the predicted class and probabilities.
    test_model(): Evaluates the trained model on the test dataset and logs detailed results.
    main(): The main entry point for the script, orchestrating the data preparation, model training, and evaluation.
Usage:
    1. Place raw Pokémon images in the `ROOT_DIR/raw` directory, organized into subdirectories by class.
    2. Run the script to preprocess the data, train the model, and evaluate its performance.
    3. Use the `predict_pokemon()` function to classify new Pokémon images.
Directory Structure:
    ROOT_DIR/
        raw/
            Charmander/
            Bulbasaur/
            Squirtle/
            Cyndaquil/
            Totodile/
            Chikorita/
Note:
    Ensure that the raw dataset is properly organized before running the script.
    The script assumes a GPU is available for training; otherwise, it falls back to CPU.
'''


import os
import shutil
import PIL
import torchvision
import torch
from torchvision import transforms
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, classification_report
# import numpy as np #XX

# Create this structure before starting
ROOT_DIR = "pokemon_data2"
CLASSES = [ "Charmander", "Bulbasaur", "Squirtle", "Cyndaquil", "Totodile", "Chikorita"]  # Add your classes

def create_directory_structure():
    for split in ["train", "val", "test"]:
        for class_name in CLASSES:
            os.makedirs(os.path.join(ROOT_DIR, split, class_name), exist_ok=True)
            
def split_dataset(raw_dir, train_transform, val_transform, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Splits the raw dataset into train, validation, and test sets.
    """
    import math
    # Ensure the ratios sum to 1
    assert math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-9), "Ratios must sum to 1.0"

    # Load the raw dataset
    raw_dataset = ImageFolder(raw_dir)

    # Calculate split sizes
    total_size = len(raw_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size  # Ensure all samples are used

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        raw_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    # Apply transforms to each split
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform

    return train_dataset, val_dataset, test_dataset

# For training data
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize all images to 128x128
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
    transforms.RandomRotation(15),  # Randomly rotate images by ±15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Randomly adjust brightness, contrast, and saturation
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Randomly translate images
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Apply random perspective transformation
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to match ResNet's expected input
])

# For validation/test data
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class PokemonClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),  # Add dropout
            nn.Linear(num_features, num_classes)
        )
        
        # Freeze initial layers
        for param in self.base_model.parameters():
            param.requires_grad = False
        # Unfreeze last few layers
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.base_model(x)
    


def train_model(model, train_loader, val_loader, num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Print statistics
        train_loss = running_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
    
    print("Training Complete!")
    return model

from PIL import Image
import imghdr #imghdr module to verify the file type

def predict_pokemon(model, image_path, class_names):
    # Check if the file is a valid image
    if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
        print(f"File does not exist or is empty: {image_path}")
        return None, None
    if imghdr.what(image_path) is None:
        print(f"Invalid image file: {image_path}")
        os.remove(image_path)  # Remove invalid file
        print(f"Removed invalid image file: {image_path}")
        return None, None

    # Load and preprocess image
    try:
        image = Image.open(image_path)
        if image.mode == "P" or image.mode == "RGBA":  # Handle palette or transparency
            image = image.convert("RGBA").convert("RGB")
        else:
            image = image.convert("RGB")
    except (PIL.UnidentifiedImageError, OSError) as e:
        print(f"Error loading image {image_path}: {e}")
        os.remove(image_path)  # Remove problematic file
        print(f"Removed problematic image file: {image_path}")
        return None, None    
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    
    # Make prediction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #xx
    model = model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    return class_names[predicted.item()], probabilities[0].cpu().numpy()

def test_model(model, test_loader, class_names):
    """
    Evaluate the model on the test dataset and log detailed results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #xx
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    print("\n--- Testing Phase ---")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Collect predictions and true labels for logging
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Calculate accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Log predictions for each batch
            for i in range(len(labels)):
                true_label = class_names[labels[i].item()]
                predicted_label = class_names[predicted[i].item()]
                probabilities = torch.nn.functional.softmax(outputs[i], dim=0).cpu().numpy()
                print(f"Image {i + 1}: True: {true_label}, Predicted: {predicted_label}, Probabilities: {probabilities}")

    # Calculate overall accuracy
    test_acc = 100 * correct / total
    print(f"\nTest Accuracy: {test_acc:.2f}%")

    # Generate and log confusion matrix #xx good for evaluation but should we use this when we turn it in?
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # Generate and log classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report:")
    print(report)

def main():
    # 1. Prepare data
    raw_dir = os.path.join(ROOT_DIR, "raw")  # Path to the raw dataset
    
    # 2. Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(
        raw_dir, train_transform, val_transform, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
    )

    # 3. Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # 4. Initialize model
    model = PokemonClassifier(num_classes=len(CLASSES))
    
    # 5. Train model
    trained_model = train_model(model, train_loader, val_loader, num_epochs=10)

    # Save the trained model
    torch.save(trained_model.state_dict(), "final_trained_model.pth")
    print("Trained model saved as 'final_trained_model.pth'")

    # 6. Test model
    test_model(trained_model, test_loader, CLASSES)

if __name__ == "__main__":
    main()