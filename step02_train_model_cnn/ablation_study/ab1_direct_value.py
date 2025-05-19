import os
import time
import shutil

import numpy as np
import torch
from torch import nn
from torch import optim

from step02_train_model_cnn.train_model import getNPY, normalize, train_and_evaluate, get_data_loader
from step02_train_model_cnn.performance_inspection import plot_cm, plot_pr, plot_roc_auc, get_predictions
from utils.plot_report import plot_report


# Setup manual cuda & np seed to ensure re-producible model output.
cuda_seed = 114514
numpy_seed = 1919810
np.random.seed(numpy_seed)
torch.cuda.manual_seed(cuda_seed)
torch.cuda.manual_seed_all(cuda_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Abl1(nn.Module):
    def __init__(self, input_channel_num, output_class_num):
        super(Abl1, self).__init__()
        self.activation = nn.SiLU()

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=1848, out_features=7392),
            self.activation,

            nn.Linear(in_features=7392, out_features=1848),
            self.activation,

            nn.Linear(in_features=1848, out_features=256),
            self.activation,

            nn.Linear(in_features=256, out_features=output_class_num)
        )

    def forward(self, x):
        x = self.fc_layers(x)
        return x


if __name__ == '__main__':
    """
    Save information
    """
    # Logging
    time_str = time.strftime("%Y%m%d-%H%M%S")
    log_root = f"./ablation_logs/{time_str}/"
    train_log_path = os.path.join(log_root, "train_log.txt")
    if os.path.exists(log_root):
        shutil.rmtree(log_root)
    os.makedirs(log_root, exist_ok=True)

    model_save_root="./ablation_models"

    dataset_source_path = "../../data/train/3dnpy"

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

    X_train_valid = X_train_valid.reshape(X_train_valid.shape[0], 2 * 7 * 12 * 11)
    X_test = X_test.reshape(X_test.shape[0], 2 * 7 * 12 * 11)

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

    # Training
    learning_rate = 2e-6
    num_epochs = 650

    # Early Stopping
    patience = 4
    min_delta = 1e-3

    model = Abl1(input_channel_num=1, output_class_num=2).to("cuda")
    criterion = nn.CrossEntropyLoss()  # Binary cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Auto adjust lr prevent o.f.

    (model,
     train_losses,
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

    model_file_name = f"posture_ablation_{time_str}.pth"
    torch.save(model_state, os.path.join(model_save_root, model_file_name))
    postamble = f"Training finished. Model saved to {os.path.join(model_save_root, model_file_name)}"
    print(postamble)

    """
    Test
    """
    model_essentials = torch.load(os.path.join(model_save_root, model_file_name))
    model = model_essentials["model_state_dict"]

    # Result Labels
    pred_scores, true_labels, pred_labels = get_predictions(ModelClass=Abl1,
                                                            model_state=model,
                                                            input_size=1,
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

