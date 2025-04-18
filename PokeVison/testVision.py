

import torch
"""
Authors: Devonte Hillman, Evan Gronewold, Svens Daukss
Overview:
    This script is designed to use the pre-trained model on a test dataset. 
    The model is based on a ResNet-50 architecture and is fine-tuned to classify images of Pokémon 

    The script performs the following tasks:
        - Loads the trained model and its weights.
        - Loads and preprocesses the test dataset.
        - Evaluates the model on the test dataset.
        - Generates a classification report and confusion matrix.
        - Visualizes sample predictions.
Resources Used:
    ChatGPT 
    DEEP SEEK 
    GEEK FOR GEEKS
    Lecture slides
    Cifar lab
    https://scikit-learn.org/stable/modules/model_evaluation.html
Classes:
    PokemonClassifier: Defines the ResNet-50-based Pokémon classifier model with a custom fully connected layer.
Functions:
    main(): The main function that orchestrates the evaluation process, including loading the model, 
    processing the test dataset, evaluating the model, and visualizing predictions.
Usage:
    1. Ensure the test dataset is organized in subdirectories by class and matches the class names defined in `CLASSES`.
    2. Update the `test_dir` variable to point to the test dataset directory.
    3. Place the trained model weights file (`final_trained_model.pth`) in the same directory as this script.
    4. Run the script using Python:
        $ python testVision.py
    5. The script will output the classification report, confusion matrix, and display sample predictions.
"""
import os
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Define the class names (must match the order of the file system)
CLASSES = [ "Bulbasaur", "Charmander", "Chikorita", "Cyndaquil", "Squirtle", "Totodile"]

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the model class (same as training)
class PokemonClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_features, num_classes))
    
    def forward(self, x):
        return self.base_model(x)

def main():
    # Load the trained model
    device = torch.device("cpu") #xx
    model = PokemonClassifier(num_classes=len(CLASSES)) 
    model.load_state_dict(torch.load("./best_model.pth", map_location=device)) # post trained model weights and architecture 
    model = model.to(device)
    model.eval() #Evaluation mode modifies the layers to suit evaluating the model such as not dropping neurons like we do when training 

    # Load test data (should be in subdirectories by class)
    test_dir = "pokemon_data2/raw"  # Update this path
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Ensure class mapping matches training
    if test_dataset.classes != CLASSES:
        raise ValueError("Test dataset classes don't match training classes")

    # Evaluation
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy for each class
    class_correct = [0] * len(CLASSES)
    class_total = [0] * len(CLASSES)

    for label, pred in zip(all_labels, all_preds):
        if label == pred:
            class_correct[label] += 1
        class_total[label] += 1

    print("Accuracy for each class RESNET18:")
    for i, class_name in enumerate(CLASSES):
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"{class_name}: {accuracy:.2f}%")

    # Visualize some predictions
    def imshow(img):
        img = img * 0.5 + 0.5  # Unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # Get some test samples
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images.cpu()
    
    # Show images
    imshow(torchvision.utils.make_grid(images[:4]))
    print("Actual:", [CLASSES[label] for label in labels[:4]])
    
    # Make predictions
    outputs = model(images[:4].to(device))
    _, predicted = torch.max(outputs, 1)
    print("Predicted:", [CLASSES[p] for p in predicted.cpu().numpy()])

if __name__ == "__main__":
    main()