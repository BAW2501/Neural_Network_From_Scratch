import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def get_mnist():
    with np.load(str(Path('./mnist.npz'))) as f:
        images, labels = f["x_train"], f["y_train"]
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels

def get_images(folder):
    return np.array([plt.imread(file).flatten() for file in folder.iterdir()])

def get_labels(folder):
    return np.array([int(file.stem.split('_')[-1]) for file in folder.iterdir()]) - 1

def load_data(path):
    return get_images(path), get_labels(path)

def get_ahcd():
    images, labels = load_data(Path('./ahcd'))
    labels = np.eye(28)[labels]
    return images, labels
