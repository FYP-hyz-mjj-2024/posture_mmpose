# Essentials
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Utilities
import os
import time

# Local
from utils.parse_file_name import parseFileName
from utils.plot_report import plot_report


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
            # class_name = npy_info['label']
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
            inputs, labels = inputs.to(device), labels.to(device)
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
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            confidence, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            print("Batch")
            for conf, pred in zip(confidence, predicted):
                print(f"    Confidence: {conf}, Prediction: {pred}")

    print(f"Accuracy: {100 * correct / total:.2f}%")


def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs=100):
    # Record Losses
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):

        # Train on Epoch
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(inputs)
        train_losses.append(running_loss/len(train_loader))

        # Evaluate one Epoch
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * len(inputs)
        test_losses.append(test_loss/len(test_loader))
        print(f"Epoch[{epoch+1}/{num_epochs}], Train Loss:{train_losses[-1]:.4f}, Test Loss:{test_losses[-1]:.4f}")

    return train_losses, test_losses


def classifyUnknown(model, input_data, thr=0.8, T=0.1):
    model.eval()
    with torch.no_grad():
        probs = model(input_data, T=T)
        # prediction = torch.argmax(probs, dim=1).item()
        max_prob = probs.max().item()
        return max_prob > thr  # True = Not Using


class ExclusiveNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ExclusiveNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, T=0.06):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return F.softmax(x / T, dim=-1)
        # return x


if __name__ == '__main__':
    data = getNPY("../data/train")
    # for key, item in data.items():
    #     print(f"{key}\n{item}\n")

    # Extract Features and Labels
    _X = []
    _y = []

    for label, data_matrix in data.items():
        _X.append(data_matrix)
        _y += [label] * len(data_matrix)

    X = np.vstack(_X)
    X[:, ::2] /= 180
    mean_X = np.mean(X)
    std_dev_X = np.std(X, ddof=1)
    X = (X - mean_X) / std_dev_X

    y = np.array(_y)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # a = np.hstack((X, np.reshape(y_encoded, (len(y_encoded), 1))))

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
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    ''' Train Model '''
    # Hyper Parameters
    input_dim = X_train.shape[1]
    hidden_dim = 128
    output_dim = len(le.classes_)

    learning_rate = 0.001
    num_epochs = 100

    # Initialize Model
    model = ExclusiveNet(input_dim, hidden_dim, output_dim).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train and Evaluate Model
    train_losses, test_losses = train_and_evaluate(model, train_loader, test_loader, loss_fn, optimizer, num_epochs)
    plot_report([train_losses, test_losses],
                ["Train Loss", "Test Loss"],
                {
                    "title": "Training Loss",
                    "x_name": "Epoch",
                    "y_name": "Loss"
                })

    model_state = {
        'model_state_dict': model.state_dict(),
        'mean_X': torch.tensor(mean_X, dtype=torch.float32),
        'std_dev_X': torch.tensor(std_dev_X, dtype=torch.float32)
    }

    torch.save(model.state_dict(), "../data/models/exclusive_nn.pth")
    print(f"Model saved to ../data/models/exclusive_nn.pth")

    """Test Unknown"""

    # X_unknown = np.load("../data/train/20240926_1509_mjj_UN-Vert_WN-Wiggle_100.npy")
    # # X_unknown = np.load("../data/train/20241003_1540_mjj_N_WN-Wiggle_000.npy")
    # # X_unknown = np.load("../data/train/20241003_1558_mjj_N_WN-FoldArms_000.npy")
    # # X_unknown = np.load("../data/train/20241003_1540_mjj_N_WN-Wiggle_000.npy")
    # # X_unknown = np.load("../data/train/20241010_1518_mjj_N_CR-Hold_000.npy")
    #
    # X_unknown[:, ::2] /= 180
    # X_unknown = (X_unknown - mean_X) / std_dev_X
    #
    # classifier = ExclusiveNet(input_dim=X_unknown.shape[1], hidden_dim=128, output_dim=len(le.classes_)).to(device)
    # classifier.load_state_dict(torch.load("../data/models/exclusive_nn.pth"))
    # classifier.eval()
    #
    # X_unknown = torch.tensor(X_unknown, dtype=torch.float32)
    #
    # test_unknown = []
    # for row in X_unknown:
    #     input_tensor = torch.tensor(row, dtype=torch.float32).to(device)
    #     not_using = classifyUnknown(classifier, input_tensor, thr=0.999, T=1.5)
    #     test_unknown.append(not_using)
    # print(f"Rate of \"Not Using\": {test_unknown.count(True)/len(test_unknown)}")
    # print(test_unknown)
