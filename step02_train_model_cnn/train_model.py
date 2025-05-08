# Basics
import os
import shutil
import copy
import time

# Essentials
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Local
from utils.parse_file_name import parseFileName
from utils.plot_report import plot_report
from step02_train_model_cnn.modules import *

if __name__ == '__main__':
    from performance_inspection import get_predictions, plot_cm, plot_pr, plot_roc_auc

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# Setup manual cuda & np seed to ensure re-producible model output.
cuda_seed = 114514
numpy_seed = 1919810
np.random.seed(numpy_seed)
torch.cuda.manual_seed(cuda_seed)
torch.cuda.manual_seed_all(cuda_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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

            # Split train_valid & test
            this_test_size = np.floor(this_npy.shape[0] * test_ratio).astype(np.int32)
            # np.random.shuffle(this_npy)
            # should not shuffle, the testing data shall be the same for each test
            this_npy_test = this_npy[:this_test_size, :]
            this_npy_train = this_npy[this_test_size:, :]

            if npy_info['label'].startswith('U'):
                npy_U_train = this_npy_train if npy_U_train is None else np.vstack((npy_U_train, this_npy_train))
                npy_U_test = this_npy_test if npy_U_test is None else np.vstack((npy_U_test, this_npy_test))

            elif npy_info['label'].startswith('N'):
                npy_N_train = this_npy_train if npy_N_train is None else np.vstack((npy_N_train, this_npy_train))
                npy_N_test = this_npy_test if npy_N_test is None else np.vstack((npy_N_test, this_npy_test))
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
    overfit_factors = []    # Deprecated.
    log_strs = [f"Training started. ID of the current model:{id(model)}"]

    # Early stopping params
    current_optimized_model = None
    current_min_valid_loss = np.inf
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
            loss = criterion(outputs, labels)
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

        # Early-stopping Mechanism
        if early_stop_params is None:
            continue

        # Update early stopping parameters
        if current_min_valid_loss - early_stop_params["min_delta"] > valid_losses[-1]:
            current_min_valid_loss = valid_losses[-1]
            current_optimized_model = copy.deepcopy(model)
            num_overfit_epochs = max(num_overfit_epochs - 1, 0)
        else:
            num_overfit_epochs += 1

        # Log the training.
        log_str = (f"Epoch[{str(epoch + 1).zfill(epoch_digit_num)}/{num_epochs}], "
                   f"Train Loss:{train_losses[-1]:.4f}, "
                   f"Valid Loss:{valid_losses[-1]:.4f}, "
                   f"OFF:{' ' if overfit_factor > 0 else ''}{overfit_factor:.4f} | "
                   f"Cur Optim: {id(current_optimized_model)}, "
                   f"Min VL: {current_min_valid_loss:.4f}, "
                   f"Num OF epochs: {num_overfit_epochs}")

        log_strs.append(log_str)
        print(log_str)

        # Perform early-stopping
        if num_overfit_epochs > early_stop_params["patience"]:
            model = current_optimized_model
            early_stop_log = f"Early stopped at epoch {epoch+1}. ID of optimized model: {id(model)}"
            log_strs.append(early_stop_log)
            print(early_stop_log)
            break

    return train_losses, valid_losses, overfit_factors, log_strs


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


def get_data_loader(inputs, labels, batch_size: int, shuffle: bool) -> DataLoader:
    """
    Obtain data loaders from inputs and labels.
    :param inputs: List of data samples.
    :param labels: List of labels.
    :param batch_size: Batch size of training.
    :param shuffle: Whether to shuffle before output.
    :return:
    """
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    tensor_dataset = TensorDataset(inputs_tensor, labels_tensor)
    loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


if __name__ == '__main__':
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
    U_train_valid, N_train_valid = train_data
    U_test, N_test = test_data

    # Get train-evaluate set and test set for both labels.
    X_train_valid = np.vstack((U_train_valid, N_train_valid))
    X_test = np.vstack((U_test, N_test))

    # Normalize train-evaluate data in per-channel manner.
    X_train_valid = normalize(X_train_valid)
    X_test = normalize(X_test)

    # Result Labels
    y_train_valid = np.hstack((np.ones(len(U_train_valid)), np.zeros(len(N_train_valid))))  # (n,)
    y_test = np.hstack((np.ones(len(U_test)), np.zeros(len(N_test))))  # (N, )

    # Get shuffle indices for train-evaluate and test.
    shuffle_indices_train_eval = np.random.permutation(X_train_valid.shape[0])

    # Shuffle and split train-evaluation data.
    X_train_valid, y_train_valid = X_train_valid[shuffle_indices_train_eval], y_train_valid[shuffle_indices_train_eval]
    split_board = int(0.35 * X_train_valid.shape[0])
    X_train, X_valid = X_train_valid[split_board:], X_train_valid[:split_board]
    y_train, y_valid = y_train_valid[split_board:], y_train_valid[:split_board]

    # Put train and evaluation data into tensor.
    train_loader = get_data_loader(X_train, y_train, batch_size=128, shuffle=True)
    valid_loader = get_data_loader(X_valid, y_valid, batch_size=128, shuffle=False)
    test_loader = get_data_loader(X_test, y_test, batch_size=32, shuffle=False)

    """
    Train
    """
    # Training
    learning_rate = 2e-6
    num_epochs = 650

    # Early Stopping
    patience = 4
    min_delta = 1e-3

    model = MLP3d(input_channel_num=2, output_class_num=2).to(device)
    criterion = nn.CrossEntropyLoss()  # Binary cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Auto adjust lr prevent o.f.

    report_loss = []

    train_message = input("Message for this train (notes, purpose, etc.): ")
    preamble = (f"Operator message: {train_message}\n"
                f"Using / Not Using: {len(U_train_valid)} / {len(N_train_valid)}\n"
                f"Max Epochs: {num_epochs}, Patience: {patience}, Min Delta: {min_delta}\n"
                f"Loss Function: {criterion.__class__.__name__}, "
                f"Learning Rate: {learning_rate}\n"
                f"CUDA seed: {cuda_seed}, NumPy seed: {numpy_seed}")
    print(preamble)

    (train_losses,
     valid_losses,
     overfit_factors,
     log_strs) = train_and_evaluate(model=model,
                                    train_loader=train_loader,
                                    valid_loader=valid_loader,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    num_epochs=num_epochs,
                                    early_stop_params={
                                        "min_delta": min_delta,
                                        "patience": patience
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
