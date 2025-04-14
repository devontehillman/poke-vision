import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #xx
    model = PokemonClassifier(num_classes=len(CLASSES)) 
    model.load_state_dict(torch.load("final_trained_model.pth", map_location=device)) # post trained model weights and architecture 
    model = model.to(device)
    model.eval() #Evaluation mode modifies the layers to suit evaluating the model such as not dropping neurons like we do when training 

    # Load test data (should be in subdirectories by class)
    test_dir = "pokemon_data2/test"  # Update this path
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

    # Generate metrics
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))
    
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

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