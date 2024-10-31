# Essentials
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split

# Utilities
import os

# Local
from utils.parse_file_name import parseFileName
from utils.plot_report import plot_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs=100):
    # Record Losses
    train_losses = []
    test_losses = []
    overfit_factors = []

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

        test_loss_last_step = np.mean(test_losses[-10:]) if epoch > 10 else 1

        overfit_factor = np.tanh(test_losses[-1]-test_loss_last_step)
        if epoch > 10:
            overfit_factors.append(overfit_factor)
        print(f"Epoch[{epoch+1}/{num_epochs}], Train Loss:{train_losses[-1]:.4f}, Test Loss:{test_losses[-1]:.4f}, "
              f"OFF:{overfit_factor:.4f}")

    return train_losses, test_losses, overfit_factors


class MLP(nn.Module):
    def __init__(self, input_channel_num, output_class_num):
        super(MLP, self).__init__()
        self.relu = nn.ELU()
        # self.input_size = input_size
        # self.output_size = output_size

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_channel_num, out_channels=8, kernel_size=3, padding=1),  # 32, 6, 268 -> 32, 8, 268
            nn.ELU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),  # 32, 8, 268 -> 32, 16, 268
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),   # 32, 16, 268 -> 32, 16, 143

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),  # 32, 16, 143 -> 32, 32, 143
            nn.ELU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),  # 32, 32, 143 -> 32, 32, 143
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 32, 32, 143 -> 32, 32, 71
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=32 * 71, out_features=256),
            nn.ELU(),
            nn.Linear(in_features=256, out_features=output_class_num)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        return x


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
    std_dev_X = np.std(X, ddof=1)
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
    initial_channel_num = 6

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_train_tensor = X_train_tensor.view(X_train.shape[0], initial_channel_num, -1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = X_test_tensor.view(X_test.shape[0], initial_channel_num, -1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Tensor Datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    """
    Model 
    """
    input_size = X_train.shape[1]
    hidden_size = 100
    learning_rate = 5e-6
    num_epochs = 650

    model = MLP(input_channel_num=initial_channel_num, output_class_num=2).to(device)
    criterion = nn.CrossEntropyLoss()    # Binary cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)    # Auto adjust lr prevent o.f.

    report_loss = []
    print(f"Start Training...\nSize of Using: {len(using)}, Size of Not Using: {len(not_using)}")

    train_losses, test_losses, overfit_factors = train_and_evaluate(model,
                                                                    train_loader,
                                                                    test_loader,
                                                                    criterion,
                                                                    optimizer,
                                                                    num_epochs)

    # Losses
    plot_report([train_losses, test_losses],
                ["Train Loss", "Test Loss"],
                {
                    "title": "TR&TE Loss w.r.t. Epoch",
                    "x_name": "Epoch",
                    "y_name": "Loss"
                })

    # Overfit Factors
    plot_report([overfit_factors],
                ["Overfit Factors"],
                {
                    "title": "Overfit Factors w.r.t. Epoch",
                    "x_name": "Epoch",
                    "y_name": "Overfit Factor"
                })

    model_state = {
        'model_state_dict': model.state_dict(),
        'mean_X': torch.tensor(mean_X, dtype=torch.float32),
        'std_dev_X': torch.tensor(std_dev_X, dtype=torch.float32)
    }

    torch.save(model_state, "../data/models/posture_mmpose_vgg.pth")
    print(f"Model saved to ../data/models/posture_mmpose_vgg.pth")



