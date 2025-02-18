# Basic
import os
import numpy as np

# Model training
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

# Model testing
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve,
                             average_precision_score)
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_predictions(ModelClass,
                    model_state,
                    test_loader: DataLoader,
                    input_size: int,
                    output_size: int):

    model = ModelClass(input_channel_num=input_size, output_class_num=output_size).to(device)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    pred_scores = []  # Prob. of predictions
    true_labels = []  # Ground Truth
    pred_labels = []  # Label of prediction, i.e., argmax(softmax(pred_scores))

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)

            # Model Inference
            outputs = model(images)

            pred_scores_batch = nn.functional.softmax(outputs, dim=-1)

            pred_scores.extend(pred_scores_batch.cpu().tolist())
            pred_labels.extend(outputs.argmax(dim=1).tolist())
            true_labels.extend(labels.cpu().tolist())

    return pred_scores, true_labels, pred_labels


def plot_cm(true_labels, pred_labels, save_path=None):
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)

    plt.title("Confusion Matrix")
    if save_path is not None:
        plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    plt.show()


def plot_pr(true_labels, pred_scores, save_path=None):
    true_labels_bin = np.array([([0, 1] if i == 1 else [1, 0])for i in true_labels])
    for i in range(0, 2):
        precision_i, recall_i, _ = precision_recall_curve(true_labels_bin[:, i], np.array(pred_scores)[:, i])
        average_precision = average_precision_score(true_labels_bin[:, i], np.array(pred_scores)[:, i])
        plt.step(recall_i, precision_i, where="post", label=f"Class {i} AP={average_precision:.2f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    if save_path is not None:
        plt.savefig(os.path.join(save_path, "precision_recall.png"))
    plt.show()


# Plot ROC AUC Curves
def plot_roc_auc(true_labels, pred_scores, save_path=None):
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
    if save_path is not None:
        plt.savefig(os.path.join(save_path, "roc_curve.png"))
    plt.show()


