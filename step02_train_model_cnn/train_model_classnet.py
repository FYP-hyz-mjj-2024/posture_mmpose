# Essentials
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Utilities
import os
import time

# Local
from utils.parse_file_name import parseFileName
from utils.plot_report import plot_report


def getNPY(npy_dir):
    labeled_data = {}

    for root, dirs, files in os.walk(npy_dir):
        for file in files:
            if not file.endswith('.npy'):
                continue

            npy_info = parseFileName(file, '.npy')

            if not npy_info['label'].startswith('U'):
                continue

            this_npy = np.load(os.path.join(root, file))
            class_name = f"{npy_info['label']}_{npy_info['extensions']}"
            if class_name not in labeled_data.keys():
                labeled_data[class_name] = this_npy
            else:
                labeled_data[class_name] = np.vstack((labeled_data[class_name], this_npy))

    return labeled_data


def train(model, train_loader, loss_fn, optimizer, num_epochs=20):
    losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        losses.append(running_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    return losses


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")


class ExclusiveNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ExclusiveNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, T=1):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        logits = self.fc3(x)
        return F.softmax(logits / T, dim=-1)


if __name__ == '__main__':
    data = getNPY("../data/train")
    # for key, item in data.items():
    #     print(f"{key}\n{item}\n")

    # Extract Features and Labels
    X = []
    y = []

    for label, data_matrix in data.items():
        X.append(data_matrix)
        y += [label] * len(data_matrix)

    X = np.vstack(X)
    y = np.array(y)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2,
                                                        random_state=1919810+int(time.time()))
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Tensor Datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create Data Loaders.
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    ''' Train Model '''
    # Hyper Parameters
    input_dim = X_train.shape[1]
    hidden_dim = 128
    output_dim = len(le.classes_)

    learning_rate = 0.001
    num_epochs = 200

    # Initialize Model
    model = ExclusiveNet(input_dim, hidden_dim, output_dim)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train Model
    report_loss = train(model, train_loader, loss_fn, optimizer, num_epochs=num_epochs)
    plot_report([report_loss],
                ["Loss"],
                {
                    "title": "Training Loss",
                    "x_name": "Epoch",
                    "y_name": "Loss"
                })

    evaluate(model, test_loader)

