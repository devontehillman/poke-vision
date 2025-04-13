import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import requests
from io import BytesIO
from torchvision.models import resnet50

# Evan Gronewold, Devonte Hillman, Svens Dauks

class PokemonDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file, skiprows=1, names=["id", "image_url", "caption", "name", "hp", "set_name"])
        self.transform = transform
        self.labels = sorted(self.data["name"].unique())
        self.label_to_idx = {name: idx for idx, name in enumerate(self.labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        url = self.data.iloc[idx]["image_url"]
        label = self.label_to_idx[self.data.iloc[idx]["name"]]

        try:
            response = requests.get(url, timeout=5)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            print(f"Error loading image: {e}")
            image = Image.new("RGB", (224, 224))  # ResNet default size

        if self.transform:
            image = self.transform(image)

        return image, label

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet input size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ResNet normalization
    ])

    batch_size = 5
    dataset = PokemonDataset(csv_file='./pokemon-cards.csv', transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = dataset.labels
    num_classes = len(classes)

    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images))
    print(' '.join(f'{classes[labels[j]]:10s}' for j in range(batch_size)))

    # Load pretrained ResNet50 and freeze all layers
    model = resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Only final layer will be trained
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print("Finished Training")
    PATH = './resnet_pokemon.pth'
    torch.save(model.state_dict(), PATH)

    # Evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the test images: {100 * correct // total} %')

    # Per-class accuracy
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
