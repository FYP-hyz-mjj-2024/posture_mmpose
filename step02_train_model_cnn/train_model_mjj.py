# Essentials
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Utilities
import os
import shutil
import copy
import time

# Local
from utils.parse_file_name import parseFileName
from utils.plot_report import plot_report

if __name__ == '__main__':
    from performance_inspection import get_predictions, plot_cm, plot_pr, plot_roc_auc

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def getNPY(npy_dir, test_ratio=0.5):
    """
    Retrieve dataset.
    :param npy_dir: Directory that stores all the .npy files, where each file is a list of features.
    :param test_ratio: Ratio of test data.
    :return: [npy_U_train, npy_N_train], [npy_U_test, npy_N_test]
    """

    if test_ratio < 0 or test_ratio > 1:
        raise ValueError("Test ratio should be between 0 and 1.")

    npy_U_train = None
    npy_N_train = None

    npy_U_test = None
    npy_N_test = None

    for root, dirs, files in os.walk(npy_dir):
        for file in files:
            if not file.endswith('.npy'):
                continue
            npy_info = parseFileName(file, '.npy')
            this_npy = np.load(os.path.join(root, file))

            # Split train & test
            this_test_size = np.floor(this_npy.shape[0] * test_ratio).astype(np.int32)
            np.random.shuffle(this_npy)
            this_npy_test = this_npy[:this_test_size, :]
            this_npy_train = this_npy[this_test_size:, :]

            if npy_info['label'].startswith('U'):
                npy_U_train = this_npy if npy_U_train is None else np.vstack((npy_U_train, this_npy_train))
                npy_U_test = this_npy if npy_U_test is None else np.vstack((npy_U_test, this_npy_test))

            elif npy_info['label'].startswith('N'):
                npy_N_train = this_npy if npy_N_train is None else np.vstack((npy_N_train, this_npy_train))
                npy_N_test = this_npy if npy_N_test is None else np.vstack((npy_N_test, this_npy_test))
            else:
                raise Exception("Wrong label retrieved.")

    return [npy_U_train, npy_N_train], [npy_U_test, npy_N_test]


def train_and_evaluate(model,
                       train_loader,
                       valid_loader,
                       criterion,
                       optimizer,
                       num_epochs=100,
                       early_stop_params=None):
    """
    Train and evaluate the posture recognition model.
    :param model: Model instance.
    :param train_loader: Train data loader.
    :param valid_loader: Validation data loader.
    :param criterion: Loss function.
    :param optimizer: Gradient descent optimizer.
    :param num_epochs: Maximum epoch number.
    :param early_stop_params: Optional dictionary of early-stop params.
    :return: Tuple of train_losses, valid_losses, overfit_factors, log_strs.
    """

    # Record Losses
    train_losses = []
    valid_losses = []
    overfit_factors = []
    log_strs = [f"Training started. ID of the current model:{id(model)}"]

    # Early stopping params
    current_optimized_model = None
    current_min_test_loss = np.inf
    num_overfit_epochs = 0

    # For display purposes only.
    epoch_digit_num = len(str(abs(num_epochs)))

    print(log_strs[0])
    for epoch in range(num_epochs):
        # Train on Epoch
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels) # TODO: May try combination of different losses, 2024-1-21 18:20
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(inputs)
        train_losses.append(running_loss / len(train_loader))

        # Validate one Epoch
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * len(inputs)
        valid_losses.append(valid_loss / len(valid_loader))

        # Overfit factor calculation
        valid_loss_last_step = np.mean(valid_losses[-10:]) if epoch > 10 else 1
        overfit_factor = np.tanh(valid_losses[-1] - valid_loss_last_step)
        if epoch > 10:
            overfit_factors.append(overfit_factor)

        # These logics are written for displaying purposes only.
        log_str = (f"Epoch[{str(epoch + 1).zfill(epoch_digit_num)}/{num_epochs}], "
                   f"Train Loss:{train_losses[-1]:.4f}, "
                   f"Valid Loss:{valid_losses[-1]:.4f}, "
                   f"OFF:{' ' if overfit_factor > 0 else ''}{overfit_factor:.4f} | "
                   f"Cur Optim: {id(current_optimized_model)}, "
                   f"Min VL: {current_min_test_loss:.4f}, "
                   f"Num OF epochs: {num_overfit_epochs}")

        log_strs.append(log_str)
        print(log_str)

        # Early-stopping Mechanism
        if early_stop_params is None:
            continue

        # Update early stopping parameters
        if current_min_test_loss - early_stop_params["min_delta"] > valid_losses[-1]:
            current_min_test_loss = valid_losses[-1]
            current_optimized_model = copy.deepcopy(model)
            num_overfit_epochs = max(num_overfit_epochs - 1, 0)
        else:
            num_overfit_epochs += 1

        # Perform early-stopping
        if num_overfit_epochs > early_stop_params["patience"]:
            model = current_optimized_model
            early_stop_log = f"Early stopped at epoch {epoch+1}. ID of optimized model: {id(model)}"
            log_strs.append(early_stop_log)
            print(early_stop_log)
            break

    return train_losses, valid_losses, overfit_factors, log_strs


class MLP3d(nn.Module):
    def __init__(self, input_channel_num, output_class_num):
        super(MLP3d, self).__init__()
        self.k = (3, 5, 5)          # Convolution3D kernel size
        self.m_k = 2                # MaxPool3D kernel stride
        self.activation = nn.ELU()

        self.conv_layers = nn.Sequential(
            # Conv 1: C=2 -> C=8
            nn.Conv3d(in_channels=input_channel_num, out_channels=8, kernel_size=self.k, padding='same'),
            nn.BatchNorm3d(num_features=8),
            self.activation,

            # Conv 2: C=8 -> C=16
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=self.k, padding='same'),
            nn.BatchNorm3d(num_features=16),
            self.activation,

            # Conv 3: C=16 -> C=32
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=self.k, padding='same'),
            nn.BatchNorm3d(num_features=32),
            self.activation,

            # Conv 4: C=32 -> C=32
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=self.k, padding='same'),
            nn.BatchNorm3d(num_features=32),
            self.activation,

            nn.MaxPool3d(kernel_size=self.m_k, stride=self.m_k),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=2880, out_features=256),
            self.activation,
            nn.Linear(in_features=256, out_features=output_class_num)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 192
        x = self.fc_layers(x)

        return x


if __name__ == '__main__':  # TODO: compatible with mode 'mjj'
    """
    Save information
    """
    # Logging
    time_str = time.strftime("%Y%m%d-%H%M%S")
    log_root = f"../logs/training_performance_log/{time_str}/"
    train_log_path = os.path.join(log_root, "train_log.txt")
    if os.path.exists(log_root):
        shutil.rmtree(log_root)
    os.makedirs(log_root, exist_ok=True)

    # Model Saving
    dataset_source_path = "../data/train/3dnpy"
    model_save_root = "../data/models/"

    """
    Prepare data
    """
    # Training data points
    train_data, test_data = getNPY(dataset_source_path, test_ratio=0.3)

    U_train, N_train = train_data # TODO: why this return form? 2024-01-21 18:11

    # Normalize Data
    # Using Z-score normalization: mean(mu)=0, std_dev(sigma)=1
    X = np.vstack((U_train, N_train))
    X[:, 0, :, :, :] /= 180.0  # Make domain of angle fields into [0, 1]
    mean_X = np.mean(X)
    std_dev_X = np.std(X, ddof=1)
    X = (X - mean_X) / std_dev_X

    # Result Labels
    y = np.hstack((np.ones(len(U_train)), np.zeros(len(N_train))))  # (n,)

    shuffle_indices = np.random.permutation(X.shape[0])

    X, y = X[shuffle_indices], y[shuffle_indices]

    split_board = int(0.35 * X.shape[0])

    # Train-test split
    X_train, X_valid = X[split_board:], X[:split_board]
    y_train, y_valid = y[split_board:], y[:split_board]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)

    # Tensor Datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)

    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

    """
    Model Training
    """
    input_size = X_train.shape[1]
    hidden_size = 100
    learning_rate = 5e-6
    num_epochs = 650

    model = MLP3d(input_channel_num=2, output_class_num=2).to(device)
    criterion = nn.CrossEntropyLoss()  # Binary cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Auto adjust lr prevent o.f.

    report_loss = []

    preamble = f"Preparing dataset...\nSize of Using: {len(U_train)}, Size of Not Using: {len(N_train)}"
    print(preamble)

    (train_losses,
     valid_losses,
     overfit_factors,
     log_strs) = train_and_evaluate(model,
                                    train_loader,
                                    valid_loader,
                                    criterion,
                                    optimizer,
                                    num_epochs,
                                    early_stop_params={
                                        "min_delta": 1e-2,
                                        "patience": 20
                                    })

    model_state = {
        'model_state_dict': model.state_dict(),
        'mean_X': torch.tensor(mean_X, dtype=torch.float32),
        'std_dev_X': torch.tensor(std_dev_X, dtype=torch.float32)
    }

    model_file_name = f"posture_mmpose_vgg3d_{time_str}.pth"
    torch.save(model_state, os.path.join(model_save_root, model_file_name))
    postamble = f"Training finished. Model saved to {os.path.join(model_save_root, model_file_name)}"
    print(postamble)

    log_strs = [preamble] + log_strs + [postamble]

    """
    Test
    """
    U_test, N_test = test_data

    model_essentials = torch.load(os.path.join(model_save_root, model_file_name))
    # model_essentials = torch.load(f"../data/models/posture_mmpose_vgg3d_17349534273325243.pth")
    model = model_essentials["model_state_dict"]
    mean = model_essentials["mean_X"].cpu().item()
    std = model_essentials["std_dev_X"].cpu().item()

    # Normalize Data
    # Using Z-score normalization: mean(mu)=0, std_dev(sigma)=1
    X_test = np.vstack((U_test, N_test))
    X_test[:, ::2] /= 180  # Make domain of angle fields into [0, 1]
    X_test = (X_test - mean) / std  # (N, C, H, W, D)

    # Result Labels
    y_test = np.hstack((np.ones(len(U_test)), np.zeros(len(N_test)))) # (N, )

    shuffle_indices_test = np.random.permutation(X_test.shape[0])
    X_test, y_test = X_test[shuffle_indices_test], y_test[shuffle_indices_test]

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Tensor Datasets
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Data Loaders
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    pred_scores, true_labels, pred_labels = get_predictions(ModelClass=MLP3d,
                                                            model_state=model,
                                                            input_size=2,
                                                            test_loader=test_loader,
                                                            output_size=2)

    """
    Plots
    """
    # Training Performances
    # Losses
    plot_report([train_losses, valid_losses],
                ["Train Loss", "Validation Loss"],
                {
                    "title": "Train and Validation Loss w.r.t. Epoch",
                    "x_name": "Epoch",
                    "y_name": "Loss"
                },
                save_path=log_root,
                file_name="epoch_loss.png")

    # Overfit Factors
    plot_report([overfit_factors],
                ["Overfit Factors"],
                {
                    "title": "Overfit Factors w.r.t. Epoch",
                    "x_name": "Epoch",
                    "y_name": "Overfit Factor"
                },
                save_path=log_root,
                file_name="overfit_factors.png")

    # Test Performances
    plot_cm(true_labels, pred_labels, save_path=log_root)
    plot_pr(true_labels, pred_scores, save_path=log_root)
    plot_roc_auc(true_labels, pred_scores, save_path=log_root)

    with open(train_log_path, "w") as f:
        f.writelines(f"{lgs}\n" for lgs in log_strs)
