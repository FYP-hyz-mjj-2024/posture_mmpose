# Basics
import os
import shutil
import copy
import time
from typing import Union, Tuple, Optional

# Essentials
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

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
            loss = criterion(outputs, labels, model) # TODO: May try combination of different losses, 2024-1-21 18:20
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


class ResPool3d(nn.Module):
    """
    Residual Pooling. Uses max-pooling to get max voxels in each local chunk.
    Then, instead of leaving others out, ResPool3d adds the max-pooled voxels
    back to the original 3d channel.
    :param kernel_size: Size of the chunk kernel of max-pool.
    :param stride: Number of steps to jump over when max-pool kernel run through.
    :param padding: Edge padding of each channel.
    :param dilation: Dilation.  (Doesn't matter here.)
    :param ceil_mode: Ceiling mode. (Doesn't matter here.)
    """

    __constants__ = ['kernel_size', 'stride', 'padding', 'dilation', 'ceil_mode']
    ceil_mode: bool

    def __init__(self,
                 kernel_size: Union[int, Tuple[int, ...]],
                 stride: Optional[Union[int, Tuple[int, ...]]] = None,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 ceil_mode: bool = False) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, input: Tensor):
        _max_pooled, indices = F.max_pool3d(input, self.kernel_size,
                                            self.stride, self.padding,
                                            self.dilation, ceil_mode=self.ceil_mode,
                                            return_indices=True)

        output_shape = input.shape

        # Initialize a blank tensor "canvas" with the same shape as input.
        output = torch.zeros(output_shape, dtype=input.dtype).to('cuda')

        # Scatter the emphasized value into the blank "canvas".
        output = output.view(-1).scatter_(0, indices.view(-1), _max_pooled.view(-1)).view(output_shape)

        # Add the emphasized values into the original input tensor.
        return input + output


class FocalLoss(nn.Module):
    """
    Alpha-balanced variant of focal loss of one epoch.
    Reference: https://arxiv.org/pdf/1708.02002
    :param alpha: Linear parameter.
    :param gamma: Exponent parameter.
    :param reduction: Reduction method: mean or sum of all samples in this batch.
    """
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if self.reduction not in ['mean', 'sum']:
            raise ValueError("Reduction method must be 'mean' or 'sum'.")

    def forward(self, inputs, labels):
        probs = torch.softmax(inputs, dim=-1)                           # get softmax of 2 classes
        pt = probs.gather(dim=-1, index=labels.unsqueeze(1))            # get p_t of groud-truth label class
        loss = (self.alpha * ((1 - pt) ** self.gamma)) * (-torch.log(pt))     # calculate focal loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class L2Regularization(nn.Module):
    """
    L2 Regularization. Prevents gradient explosion.
    :param l2_lambda: Lambda of the L2 regularization, penalize factor.
    """
    def __init__(self, l2_lambda=0.0001):
        super(L2Regularization, self).__init__()
        self.l2_lambda = l2_lambda

    def forward(self, _model):
        _device = next(_model.parameters()).device

        # Loss: Square sum of all weights.
        l2_reg = torch.tensor(0., device=_device)
        for param in _model.parameters():
            l2_reg += torch.sum(param ** 2)

        return self.l2_lambda * l2_reg


class MCLoss(nn.Module):
    """
    MCLoss stands for Mean Cross-entropy loss. It is a weighted combination of
    BCE Loss, Focal Loss and L2 Regularization.
    """
    def __init__(self, w1=0.6, w2=0.3, w3=0.1, focal_alpha=0.25, focal_gamma=2, l2_lambda=0.0001):
        super(MCLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.focal = FocalLoss(focal_alpha, focal_gamma)
        self.l2 = L2Regularization(l2_lambda)

    def forward(self, outputs, labels, mlp3d_instance):
        # BCE with logits
        unsoftmax_pt = outputs.gather(dim=-1, index=labels.unsqueeze(1))    # (batch_size=128, 1)
        input_ = unsoftmax_pt.squeeze(1)        # (batch_size=128, ) array of ground-truth label prob.
        target_ = labels.to(torch.float32)      # Ground-truth labels
        bce_loss = F.binary_cross_entropy_with_logits(input_, target_)

        # Focal Loss
        focal_loss = self.focal(outputs, labels)

        # L2 Regularization
        l2_reg = self.l2(mlp3d_instance)

        # Total Loss
        total_loss = self.w1 * bce_loss + self.w2 * focal_loss + self.w3 * l2_reg
        return total_loss


class MLP3d(nn.Module):
    def __init__(self, input_channel_num, output_class_num):
        super(MLP3d, self).__init__()
        self.k = (3, 5, 5)          # Convolution3D kernel size
        self.m_k = 2                # MaxPool3D kernel stride
        self.activation = nn.SiLU()

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

            ResPool3d(kernel_size=self.m_k, stride=self.m_k, padding=0),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=29568, out_features=7392),
            self.activation,

            nn.Linear(in_features=7392, out_features=1848),
            self.activation,

            nn.Linear(in_features=1848, out_features=256),
            self.activation,

            nn.Linear(in_features=256, out_features=output_class_num)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 29568
        x = self.fc_layers(x)
        return x


def normalize(X):
    """
    Normalize data.
    :param X: 5-D input data. (N, C, D, H, W).
    :return:
    """
    mean_angles = np.mean(X[:, 0, :, :, :])  # No need to divide by 180 since channels are separated.
    mean_scores = np.mean(X[:, 1, :, :, :])

    std_angles = np.std(X[:, 0, :, :, :])
    std_scores = np.std(X[:, 1, :, :, :])

    X[:, 0, :, :, :] = (X[:, 0, :, :, :] - mean_angles) / std_angles
    X[:, 1, :, :, :] = (X[:, 1, :, :, :] - mean_scores) / std_scores

    return X


if __name__ == '__main__':  # TODO: compatible with mode 'mjj'
    """
    Save information
    """
    # Logging
    time_str = time.strftime("%Y%m%d-%H%M%S")
    log_root = f"./logs/{time_str}/"
    train_log_path = os.path.join(log_root, "train_log.txt")
    if os.path.exists(log_root):
        shutil.rmtree(log_root)
    os.makedirs(log_root, exist_ok=True)

    # Model Saving
    dataset_source_path = "../data/train/3dnpy"
    model_save_root = "./archived_models/"

    """
    Prepare data
    """
    # Training data points
    train_data, test_data = getNPY(dataset_source_path, test_ratio=0.3)
    U_train, N_train = train_data
    U_test, N_test = test_data

    # Get train-evaluate set and test set for both labels.
    X_train_eval = np.vstack((U_train, N_train))
    X_test = np.vstack((U_test, N_test))

    # Normalize train-evaluate data in per-channel manner.
    X_train_eval = normalize(X_train_eval)
    X_test = normalize(X_test)

    # Result Labels
    y = np.hstack((np.ones(len(U_train)), np.zeros(len(N_train))))  # (n,)
    y_test = np.hstack((np.ones(len(U_test)), np.zeros(len(N_test))))  # (N, )

    # Get shuffle indices for train-evaluate and test.
    shuffle_indices_train_eval = np.random.permutation(X_train_eval.shape[0])
    # shuffle_indices_test = np.random.permutation(X_test.shape[0])

    # Shuffle and split train-evaluation data.
    X_train_eval, y = X_train_eval[shuffle_indices_train_eval], y[shuffle_indices_train_eval]
    split_board = int(0.35 * X_train_eval.shape[0])
    X_train, X_valid = X_train_eval[split_board:], X_train_eval[:split_board]
    y_train, y_valid = y[split_board:], y[:split_board]
    # X_test, y_test = X_test[shuffle_indices_test], y_test[shuffle_indices_test]

    # Put train and evaluation data into tensor.
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Tensor Datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    """
    Train
    """
    learning_rate = 5e-6
    num_epochs = 650

    model = MLP3d(input_channel_num=2, output_class_num=2).to(device)
    # criterion = nn.CrossEntropyLoss()  # Binary cross entropy loss
    criterion = MCLoss(0.6, 0.3, 0.1)  # Binary cross entropy loss
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
                                        "min_delta": 1e-3,
                                        "patience": 8
                                    })

    model_state = {
        'model_state_dict': model.state_dict(),
    }

    model_file_name = f"posture_mmpose_vgg3d_{time_str}.pth"
    torch.save(model_state, os.path.join(model_save_root, model_file_name))
    postamble = f"Training finished. Model saved to {os.path.join(model_save_root, model_file_name)}"
    print(postamble)

    log_strs = [preamble] + log_strs + [postamble]

    """
    Test
    """
    model_essentials = torch.load(os.path.join(model_save_root, model_file_name))
    model = model_essentials["model_state_dict"]

    # Result Labels
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
