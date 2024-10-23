import numpy as np

npy = np.load("../data/train/20240926_1509_mjj_UN-Vert_WN-Wiggle_100.npy")
scores = npy[:, ::-2]

a = 1