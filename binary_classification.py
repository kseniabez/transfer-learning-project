import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

data_root = "datasets"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for ResNet18:
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def train_network(network, loader, criterion, optimizer, n_epochs):
    network.train()

    for epoch in range(n_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, Y_batch in loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()
            S = network(X_batch)
            loss = criterion(S, Y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, P = torch.max(S, 1)
            correct += (P == Y_batch).sum().item()
            total += Y_batch.size(0)

        acc = 100 * correct / total
        print(f"[Epoch {epoch+1}/{n_epochs}] Loss: {running_loss:.4f} | Train Accuracy: {acc:.2f}%")

def compute_accuracy(network, loader):
    network.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            S = network(X_batch)
            _, P = torch.max(S, 1)
            correct += (P == Y_batch).sum().item()
            total += Y_batch.size(0)

    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")


if __name__ == '__main__':
    train_data = datasets.OxfordIIITPet(
        root=data_root,
        split='trainval',
        target_types='binary-category',
        transform=transform,
        download=False
    )
    test_data = datasets.OxfordIIITPet(
        root=data_root,
        split='test',
        target_types='binary-category',
        transform=transform,
        download=False
    )

    # Data loaders
    batch_size = 32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # ResNet18
    network = models.resnet18(weights='DEFAULT')
    nf = network.fc.in_features
    network.fc = nn.Linear(nf, 2)
    network = network.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=1e-4)

    train_network(network, train_loader, criterion, optimizer, 3)
    compute_accuracy(network, test_loader)

