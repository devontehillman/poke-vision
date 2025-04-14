import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import requests
from io import BytesIO

# Evan Gronewold, Devonte Hillman, Svens Dauks

target_pokemon = ["Charmander", "Bulbasaur", "Squirtle", "Cyndaquil", "Totodile", "Chikorita"] # gen2 starters 

class PokemonDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file, skiprows=1, names=["id", "image_url", "caption", "name", "hp", "set_name"]) # use pandas to read in the csv, skip 1st row 
        # since that row is header
        # Filter only rows where 'name' is in target_pokemon, so pikachu charmander etc
        self.data = self.data[self.data['name'].isin(target_pokemon)] # reset the index to avoid duplicate pokemon
        self.data.to_csv('filtered_pokemon.csv', index=False)  # Save the filtered DataFrame to a CSV file
        print(self.data.head())  # Displays the first 5 rows of the DataFrame
        exit()  # Stops the program
        self.transform = transform
        self.labels = sorted(self.data["name"].unique())  # build label list
        self.label_to_idx = {name: idx for idx, name in enumerate(self.labels)} # map names to indices here 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the image from the image url 
        url = self.data.iloc[idx]["image_url"]
        label = self.label_to_idx[self.data.iloc[idx]["name"]]

        try:
            response = requests.get(url, timeout=5) # request the link and set timout so code can run smoother on laptop 
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            print(f"Error loading image: {e}")
            image = Image.new("RGB", (50, 50))  # fallback blank image

        if self.transform:
            image = self.transform(image)

        return image, label

# Everything here from lab with small changes to fit pokemon set 
def main():
    transform = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # changed off grescale to RGB 
    ])
    
    transformHoriz = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # changed off grescale to RGB 
    ])
    
    transformVert = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.RandomVerticalFlip(p=0.5),  # Randomly flip images horizontally
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # changed off grescale to RGB 
    ])
    transformFilter = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Add random color jitter
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # changed off grescale to RGB 
    ])

    batch_size = 6

    dataset = PokemonDataset(csv_file='./pokemon-cards.csv', transform=transform)
    
    # balanced_data = pd.concat([
    # dataset.data[dataset.data["name"] == name].sample(n=100, replace=True) # got too many sample of pikachu and was overfitting 
    # for name in target_pokemon
    # ]).reset_index(drop=True)
    # dataset.data = balanced_data

    train_size = int(0.8 * len(dataset)) # 80% of data used for training
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = dataset.labels  # class names = Pokémon names
    num_classes = len(classes) # number of classes for pokemon names


    # Helper function to display the image
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # Get some random training images that can be displayed 
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Show images with imshow 
    imshow(torchvision.utils.make_grid(images))
    # Print labels
    print(' '.join(f'{classes[labels[j]]:10s}' for j in range(batch_size)))

    class Net(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 9 * 9, 120) # had to manually change this since resizeing the image to 50x50 to see if that got better results
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        
    net = Net(num_classes)

    # loss function and use momentum with stochastic gradient descent 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10): # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999: # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print("Finished Training")

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Show images
    imshow(torchvision.utils.make_grid(images))

    net = Net(num_classes)
    net.load_state_dict(torch.load(PATH))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:10s}' for j in range(batch_size)))


    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(f'Accuracy of the network on the {length} test images: {100 * correct // total} %')


    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')



    saveLocation = './cifar_net.pth'
    net.load_state_dict(torch.load(saveLocation))
    print(dataset.data['name'].value_counts())
                        
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()