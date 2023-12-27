
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train(net, trainloader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            # images, labels = images.to(DEVICE), labels.to(DEVICE)
            # loss = criterion(net(images), labels)
            # loss.backward()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()
                        
def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images.to(DEVICE))
            loss += criterion(outputs, labels.to(DEVICE)).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(testloader.dataset) , correct / total
            
    #     for data in testloader:
    #         images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
    #         outputs = net(images)
    #         loss += criterion(outputs, labels).item()
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    # accuracy = correct / total
    # return loss, accuracy
    
def load_data():
    """Load CIFAR-10 (training and test set)."""
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform = transforms.Compose(
    # [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )
    trainset = CIFAR10("./data", train=True, download=True, transform=transform)
    testset = CIFAR10("./data", train=False, download=True, transform=transform)
    # trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    # testloader = DataLoader(testset, batch_size=32)
    # num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
    # return trainloader, testloader, num_examples
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)

def load_model():
    return Net().to(DEVICE)

if __name__ == "__main__":
    net = load_model()
    print("Load data")
    trainloader, testloader = load_data()
    print("Start training")
    train(net=net, trainloader=trainloader, epochs=5)
    print("Evaluate model")
    loss, accuracy = test(net=net, testloader=testloader)
    print(f"Loss: {loss:.5f}, Accuracy: {accuracy:.3f}")
    
    # print("Centralized PyTorch training")
    # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # # Load model and data
    # print("Load data")
    # trainloader, testloader, num_examples = load_data()
    # print("Start training")
    # net = Net().to(DEVICE)
    # train(net=net, trainloader=trainloader, epochs=2, device=DEVICE)
    # print("Evaluate model")
    # loss, accuracy = test(net=net, testloader=testloader, device=DEVICE)
    # print(f"Loss: {loss:.5f}, Accuracy: {accuracy:.3f}")