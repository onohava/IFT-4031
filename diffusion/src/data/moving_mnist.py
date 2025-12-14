import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class MovingMNISTDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        num_frames: int = 16,
        image_size: int = 64,
        train: bool = True,
        transform=None,
    ):
        self.data_path = Path(data_path)
        self.num_frames = num_frames
        self.image_size = image_size
        self.train = train
        self.transform = transform

        self.data = self._load_data()

    def _load_data(self) -> np.ndarray:
        data_file = self.data_path / "mnist_test_seq.npy"

        if not data_file.exists():
            raise FileNotFoundError(
                f"MovingMNIST data not found at {data_file}. "
                "Download from http://www.cs.toronto.edu/~nitish/unsupervised_video/"
            )

        data = np.load(data_file)
        data = np.transpose(data, (1, 0, 2, 3))

        split_idx = int(len(data) * 0.8)
        if self.train:
            data = data[:split_idx]
        else:
            data = data[split_idx:]

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.data[idx, :self.num_frames]

        seq = seq.astype(np.float32) / 255.0
        seq = seq * 2 - 1

        seq = seq[:, np.newaxis, :, :]
        seq = torch.from_numpy(seq)

        if self.transform:
            seq = self.transform(seq)

        return seq


def create_moving_mnist_dataloader(
    data_path: str,
    batch_size: int = 32,
    num_frames: int = 16,
    train: bool = True,
    num_workers: int = 4,
):
    from torch.utils.data import DataLoader

    dataset = MovingMNISTDataset(
        data_path=data_path,
        num_frames=num_frames,
        train=train,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
    )
