# dataset.py
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.io import read_video
import os
from urllib.request import urlretrieve
import random

class MovingMNIST(Dataset):
    def __init__(self, path):
        self.path = path

        if not os.path.exists(self.path):
            print("Downloading Moving MNIST...")
            urlretrieve("http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy", self.path)

        self.data = np.load(self.path)  # Shape: (20, 10000, 64, 64)
        self.data = self.data.transpose(1, 0, 2, 3)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        seq = self.data[idx]

        seq = seq.astype(np.float32) / 255.0
        return torch.from_numpy(seq).unsqueeze(1)


class UCF101Transform:
    """
    Optimized transform for video tensors.
    Input: (T, H, W, C) uint8 tensor from read_video
    Output: (C, T, H, W) float tensor normalized [0,1] and grayscaled
    """
    def __init__(self, size=(64, 64)):
        self.size = size

    def __call__(self, video):
        video = video.permute(0, 3, 1, 2)
        video = video.float() / 255.0

        video = T.Grayscale(num_output_channels=1)(video)
        video = T.Resize(self.size)(video)
        return video


class ActionVideoDataset(Dataset):
    def __init__(
            self,
            root,
            actions=None,  # List of strings, e.g., ["WalkingWithDog"]
            frames_per_clip=16,
            step_between_clips=1,
            train=True
    ):
        self.root = Path(root)
        self.frames_per_clip = frames_per_clip
        self.transform = UCF101Transform()

        all_classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])

        if actions:
            self.classes = [c for c in all_classes if c in actions]
        else:
            self.classes = all_classes

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 3. Collect all video files
        self.samples = []
        for cls_name in self.classes:
            cls_folder = self.root / cls_name
            # Extensions can be adjusted (.avi, .mp4, etc.)
            videos = list(cls_folder.glob("*.avi")) + list(cls_folder.glob("*.mp4"))
            for video_path in videos:
                self.samples.append((str(video_path), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        video, audio, info = read_video(path, pts_unit='sec')
        total_frames = video.shape[0]

        # Temporal Random Crop (or center crop)
        if total_frames > self.frames_per_clip:
            start = random.randint(0, total_frames - self.frames_per_clip)
            video = video[start: start + self.frames_per_clip]
        else:
            # Padding logic if video is too short (optional, here we just loop it or repeat last frame)
            # Simple approach: repeat video until it fits
            while video.shape[0] < self.frames_per_clip:
                video = torch.cat([video, video], dim=0)
            video = video[:self.frames_per_clip]

        # Apply optimized transforms
        if self.transform:
            video = self.transform(video)

        return video


def get_dataloader(cfg):
    if cfg.DATASET_NAME == 'MovingMNIST':
        # Assuming MovingMNIST is defined elsewhere
        data = MovingMNIST("data/mnist_test_seq.npy")
    else:
        # Use the new custom class
        data = ActionVideoDataset(
            root="data/UCF-101",
            actions=["JumpRope", "WalkingWithDog"],
            frames_per_clip=16
        )

    train_loader = DataLoader(
        data,
        batch_size=getattr(cfg, 'BATCH_SIZE', 32),
        shuffle=True,
        pin_memory=True
    )
    return train_loader