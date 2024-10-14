# Essentials
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

# Utilities
import os

# Local
from utils.parse_file_name import parseFileName
from utils.plot_report import plot_report


def getNPY(npy_dir):
    npy_using = None
    npy_not_using = None

    for root, dirs, files in os.walk(npy_dir):
        for file in files:
            if not file.endswith('.npy'):
                continue
            npy_info = parseFileName(file, '.npy')
            this_npy = np.load(os.path.join(root,file))
            if npy_info['label'].startswith('U'):
                npy_using = this_npy if npy_using is None else np.vstack((npy_using,this_npy))
            elif npy_info['label'].startswith('N'):
                npy_not_using = this_npy if npy_not_using is None else np.vstack((npy_not_using,this_npy))
            else:
                raise Exception("Wrong label retrieved.")

    return npy_using, npy_not_using


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.relu = nn.ELU()

        # Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        # Fully-connected Layers
        self.fc1 = nn.Linear(64, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #
        x = x.unsqueeze(2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)

        # Expand
        x = x.view(x.size(0), -1)

        # Forward through fc layers
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        x = self.fc4(x)
        return torch.sigmoid(x)


if __name__ == '__main__':
    """
    Prepare data
    """
    # Training data points
    using, not_using = getNPY("../data/train")

    # Normalize Data
    # Using Z-score normalization: mean(mu)=0, std_dev(sigma)=1
    X = np.vstack((using, not_using))
    X[:, ::2] /= 180    # Make domain of angle fields into [0, 1]
    mean_X = np.mean(X)
    std_dev_X = np.std(X)
    X = (X - mean_X) / std_dev_X

    # Result Labels
    y = np.hstack((np.ones(len(using)), np.zeros(len(not_using))))

    # Horizontal Concatenate
    X_y = np.hstack((X, y.reshape(1, len(y)).T))

    # Shuffle Matrix
    np.random.shuffle(X_y)

    # Train-test split
    X_y_train, X_y_test = train_test_split(X_y, test_size=0.35, random_state=114514)
    X_train, y_train = X_y_train[:, :-1], X_y_train[:, -1]
    X_test, y_test = X_y_test[:, :-1], X_y_test[:, -1]

    # Put into torch tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    """
    Model 
    """
    input_size = X_train.shape[1]
    hidden_size = 100
    learning_rate = 0.0001
    num_epochs = 500

    model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=2)
    criterion = nn.CrossEntropyLoss()    # Binary cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)    # Auto adjust lr prevent o.f.

    report_loss = []
    print(f"Start Training...\nSize of Using: {len(using)}, Size of Not Using: {len(not_using)}")
    for epoch in range(num_epochs):
        model.train()

        # Forward pass
        logits = model(X_train_tensor)      # Shape=(num_people, 2), where 2 is the two probs of "using" & "not using"
        loss = criterion(logits, y_train_tensor)

        # Backward pass, optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Report
        report_loss.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    plot_report(
        [report_loss],
        ['Loss'],
        {
            'title': 'Training Loss',
            'x_name': 'Epoch',
            'y_name': 'Loss'
        })

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_tensor)
        train_predicted = torch.argmax(train_outputs, dim=1)
        train_accuracy = (train_predicted == y_train_tensor).float().mean()

        test_outputs = model(X_test_tensor)
        test_predicted = torch.argmax(test_outputs, dim=1)
        test_accuracy = (test_predicted == y_test_tensor).float().mean()

    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    torch.save(model.state_dict(), "../data/models/posture_mmpose_nn.pth")
    print(f"Model saved to ../data/models/posture_mmpose_nn.pth")



