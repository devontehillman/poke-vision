import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Define the class names (must match the order used during training)
CLASSES = ["Bulbasaur", "Charmander", "Chikorita", "Cyndaquil", "Squirtle", "Totodile"]

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

def predict_pokemon(image_path, model, transform):
    """Predict the Pokémon in a single image"""
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Move to device and make prediction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # Get the predicted class and confidence
    predicted_class = CLASSES[predicted.item()]
    confidence = probabilities[0][predicted.item()].item()
    
    # Display the image
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_class} ({confidence:.2%})")
    plt.show()
    
    return predicted_class, confidence

def main():
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PokemonClassifier(num_classes=len(CLASSES))
    model.load_state_dict(torch.load("final_trained_model.pth", map_location=device))
    model = model.to(device)
    model.eval()

    # Example usage - replace with your image path
    image_path = "./oneImage/poke2.jpg"  
    
    # Make prediction
    pokemon, confidence = predict_pokemon(image_path, model, transform)
    print(f"Predicted Pokémon: {pokemon} (Confidence: {confidence:.2%})")

if __name__ == "__main__":
    main()