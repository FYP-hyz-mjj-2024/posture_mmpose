import torch
import numpy as np
from tqdm import tqdm
# from typing import override
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from step02_train_model_cnn.train_model import getNPY, normalize, get_data_loader
from step02_train_model_cnn.modules import MLP3d
from step02_train_model_cnn.ablation_study.ab1_direct_value import Abl1


class MLP3dEval(MLP3d):
    def __init__(self, input_channel_num, output_class_num):
        super(MLP3dEval, self).__init__(input_channel_num, output_class_num)

    # @override
    def forward(self, x):
        x = self.conv_layers(x)
        x_flat = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x_flat = self.fc_layers[0](x_flat)
        x_flat = self.fc_layers[1](x_flat)
        x_flat = self.fc_layers[2](x_flat)
        return x_flat  # Return the final output for evaluation


class Abl1Eval(Abl1):
    def __init__(self, input_channel_num, output_class_num):
        super(Abl1Eval, self).__init__(input_channel_num, output_class_num)

    def forward(self, x):
        x_flat = self.fc_layers[0](x)
        x_flat = self.fc_layers[1](x_flat)
        x_flat = self.fc_layers[2](x_flat)
        return x_flat


def tsne_analysis(_data, _labels, title):
    """
    Perform a t-SNE analysis on the data.
    :param _data:
    :param _labels:
    :return:
    """

    if _data.shape[0] != len(_labels):
        raise Exception("Data and labels should be the same length.")

    tsne = TSNE(n_components=2, random_state=42)
    data_3d = tsne.fit_transform(_data)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], s=0.1, c=_labels, cmap="jet")
    ax.set_title(title)
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    plt.colorbar(scatter, label="Label")
    plt.show()


def pca_analysis(_data, _labels, title):
    """
    Perform a principal component analysis on the data.
    :param _data:
    :param _labels:
    :return:
    """

    if _data.shape[0] != len(_labels):
        raise Exception("Data and labels should be the same length.")

    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(_data)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], s=0.1, c=_labels, cmap="jet")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(scatter, label="Label")
    plt.show()


if __name__ == '__main__':
    conv_model_essentials = torch.load("../archived_models/posture_mmpose_vgg3d_20250508-132048.pth")
    conv_model_state = conv_model_essentials["model_state_dict"]

    fc_model_essentials = torch.load("./ablation_models/posture_ablation_20250512-170252.pth")
    fc_model_state = fc_model_essentials["model_state_dict"]

    [data_U, data_N], _ = getNPY("../../data/train/3dnpy", test_ratio=0)

    data_U = normalize(data_U)
    data_N = normalize(data_N)

    X = np.vstack((data_U, data_N))
    y = np.hstack((np.ones(len(data_U)), np.zeros(len(data_N))))

    data_loader = get_data_loader(X, y, batch_size=32, shuffle=False)
    conv_model = MLP3dEval(input_channel_num=2, output_class_num=2).to("cuda")
    conv_model.load_state_dict(conv_model_state)
    conv_model.to("cuda")
    conv_model.eval()

    fc_model = Abl1Eval(input_channel_num=1, output_class_num=2).to("cuda")
    fc_model.load_state_dict(fc_model_state)
    fc_model.to("cuda")
    fc_model.eval()

    original_data = []
    conv_data = []
    fc_data = []
    labels = []

    with torch.no_grad():
        for _2c3ds, batch_labels in tqdm(data_loader):
            labels.extend(batch_labels)
            flattened_data = _2c3ds.reshape(_2c3ds.shape[0], 2 * 7 * 12 * 11).to("cuda")
            original_data.extend(flattened_data.tolist())
            _2c3ds = _2c3ds.to("cuda")

            conv_output = conv_model(_2c3ds)
            fc_output = fc_model(flattened_data)

            conv_output_cpu = conv_output.detach().cpu().tolist()
            fc_output_cpu = fc_output.detach().cpu().tolist()

            conv_data.extend(conv_output_cpu)
            fc_data.extend(fc_output_cpu)

    original_data = np.array(original_data)
    conv_data = np.array(conv_data)
    fc_data = np.array(fc_data)

    # print(convoluted_data.shape)
    print(f"Conducting t-SNE analysis on flattened original data.")
    tsne_analysis(_data=original_data, _labels=labels, title="t-SNE Analysis on Flattened Original Data")

    print(f"Conducting t-SNE analysis on hijacked conv data.")
    tsne_analysis(_data=conv_data, _labels=labels, title="t-SNE Analysis on Hijacked Conv Data")

    print(f"Conducting t-SNE analysis on hijacked fc data.")
    tsne_analysis(_data=fc_data, _labels=labels, title="t-SNE Analysis on Hijacked FC Data")








