import torch
import torch.nn as nn
from tqdm import tqdm
from train_model_hyz import MLP, getNPY
from torch.utils.data import Dataset, DataLoader, TensorDataset

import numpy as np

from sklearn.metrics import (confusion_matrix, accuracy_score,
                            precision_score, recall_score,
                            f1_score, roc_auc_score,
                            roc_curve, auc, precision_recall_curve,
                            average_precision_score)
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_predictions(model_path, extra_loader, input_size, hidden_size, output_size):
    if not isinstance(model_path, str):
        model_state = model_path
    else:
        model_state = torch.load(model_path)
    model = MLP(input_channel_num=input_size, hidden_size=hidden_size, output_class_num=output_size).to(device)
    model.load_state_dict(model_state)

    model.to(device)
    model.eval()

    pred_scores = []  # Prob. of predictions
    true_labels = []  # Ground Truth
    pred_labels = []  # Label of prediction, i.e., argmax(softmax(pred_scores))

    with torch.no_grad():
        for images, labels in tqdm(extra_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            pred_scores_batch = nn.functional.softmax(outputs, dim=-1)

            pred_scores.extend(pred_scores_batch.cpu().tolist())
            pred_labels.extend(outputs.argmax(dim=1).tolist())
            true_labels.extend(labels.cpu().tolist())

    return pred_scores, true_labels, pred_labels


def plot_cm(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


def plot_pr(true_labels, pred_labels):
    true_labels_bin = np.array([([0, 1] if i == 1 else [1, 0])for i in true_labels])
    for i in range(0, 2):
        precision_i, recall_i, _ = precision_recall_curve(true_labels_bin[:, i], np.array(pred_scores)[:, i])
        average_precision = average_precision_score(true_labels_bin[:, i], np.array(pred_scores)[:, i])
        plt.step(recall_i, precision_i, where="post", label=f"Class {i} AP={average_precision:.2f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    plt.show()


# Plot ROC AUC Curves
def plot_roc_auc(true_labels, pred_scores):
    # Compute ROC AUC for each class
    def get_roc_auc(true_labels_bin, pred_scores):
        roc_auc = dict()
        for i in range(0, 2):
            roc_auc[i] = roc_auc_score(true_labels_bin[:, i], np.array(pred_scores)[:, i])
        return roc_auc

    true_labels_bin = np.array([([0, 1] if i == 1 else [1, 0])for i in true_labels])
    roc_auc = get_roc_auc(true_labels_bin, pred_scores)

    for i in range(0, 2):
        fpr, tpr, _ = roc_curve(true_labels_bin[:, i], np.array(pred_scores)[:, i])
        plt.plot(fpr, tpr, label=f"Class {i}, AUC={roc_auc[i]:.2f}")

    plt.plot([0, 1], [0, 1], "k--")  # Diagnal Line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # plt.yscale("logit")
    plt.title("ROC Curve")
    plt.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    using, not_using = getNPY("../data/train")

    model_essentials = torch.load("../data/models/posture_mmpose_nn.pth")
    model = model_essentials["model_state_dict"]
    mean = model_essentials["mean_X"].cpu().item()
    std = model_essentials["std_dev_X"].cpu().item()

    # Normalize Data
    # Using Z-score normalization: mean(mu)=0, std_dev(sigma)=1
    X = np.vstack((using, not_using))
    X[:, ::2] /= 180  # Make domain of angle fields into [0, 1]
    X = (X - mean) / std

    # Result Labels
    y = np.hstack((np.ones(len(using)), np.zeros(len(not_using))))

    # Horizontal Concatenate
    X_y = np.hstack((X, y.reshape(1, len(y)).T))

    # Shuffle Matrix
    np.random.shuffle(X_y)

    X = X_y[:, :-1]
    y = X_y[:, -1]

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Tensor Datasets
    all_dataset = TensorDataset(X_tensor, y_tensor)

    # Data Loaders
    all_loader = DataLoader(all_dataset, batch_size=32, shuffle=True)

    pred_scores, true_labels, pred_labels = get_predictions(model, input_size=X.shape[1],extra_loader=all_loader, hidden_size=100, output_size=2)
    # print(pred_scores)

    plot_cm(true_labels, pred_labels)
    plot_pr(true_labels, pred_scores)
    plot_roc_auc(true_labels, pred_scores)




