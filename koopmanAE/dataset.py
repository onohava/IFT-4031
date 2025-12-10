# dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from urllib.request import urlretrieve
from config import Config

class MovingMNIST(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        self.file_path = os.path.join(root, 'mnist_test_seq.npy')

        if not os.path.exists(root):
            os.makedirs(root)
        if not os.path.exists(self.file_path):
            print("Downloading Moving MNIST...")
            urlretrieve("http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy", self.file_path)

        self.data = np.load(self.file_path)  # Shape: (20, 10000, 64, 64)

        if train:
            self.data = self.data[:, :9000, ...]
        else:
            self.data = self.data[:, 9000:, ...]

        self.data = self.data.transpose(1, 0, 2, 3)  # (N, 20, 64, 64)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        seq = self.data[idx]

        seq = seq.astype(np.float32) / 255.0
        return torch.from_numpy(seq).unsqueeze(1)


def get_dataloader(cfg):
    train_set = MovingMNIST(cfg.DATA_PATH, train=True)
    train_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
    return train_loader, None