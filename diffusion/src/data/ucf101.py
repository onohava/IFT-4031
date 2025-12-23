"""
UCF101 Dataset for Video Diffusion Training.

Uses pre-extracted frames from /datasets01/UCF101_Frames/frames/
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms


class UCF101FramesDataset(Dataset):
    """
    UCF101 dataset loading pre-extracted frames.

    Args:
        root: Path to frames directory (e.g., /datasets01/UCF101_Frames/frames)
        actions: List of action classes to include (e.g., ["ApplyLipstick"])
                 If None, uses all classes
        num_frames: Number of frames per clip
        image_size: Target image size (frames are resized to this)
        train: If True, use first 80% of videos; else last 20%
        grayscale: If True, convert to grayscale
    """

    def __init__(
        self,
        root: str,
        actions: list = None,
        num_frames: int = 16,
        image_size: int = 64,
        train: bool = True,
        grayscale: bool = True,
    ):
        self.root = Path(root)
        self.num_frames = num_frames
        self.image_size = image_size
        self.train = train
        self.grayscale = grayscale

        # Build transform
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
        if grayscale:
            transform_list.insert(0, transforms.Grayscale(num_output_channels=1))
        self.transform = transforms.Compose(transform_list)

        # Get action classes
        if actions is None:
            self.actions = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        else:
            self.actions = actions

        # Collect video folders
        self.videos = []
        for action in self.actions:
            action_dir = self.root / action
            if action_dir.exists():
                video_dirs = sorted([d for d in action_dir.iterdir() if d.is_dir()])
                # Train/test split
                split_idx = int(len(video_dirs) * 0.8)
                if train:
                    video_dirs = video_dirs[:split_idx]
                else:
                    video_dirs = video_dirs[split_idx:]
                self.videos.extend(video_dirs)

        print(f"UCF101: {len(self.videos)} videos from {len(self.actions)} action(s), "
              f"{'train' if train else 'test'} split")

    def __len__(self) -> int:
        return len(self.videos)

    def _load_frames(self, video_dir: Path) -> torch.Tensor:
        """Load frames from a video directory."""
        # Get sorted frame files
        frame_files = sorted(video_dir.glob("*.jpg"))
        if len(frame_files) == 0:
            frame_files = sorted(video_dir.glob("*.png"))

        total_frames = len(frame_files)

        if total_frames < self.num_frames:
            # Repeat frames if video is too short
            indices = list(range(total_frames))
            while len(indices) < self.num_frames:
                indices.extend(list(range(total_frames)))
            indices = indices[:self.num_frames]
        else:
            # Random temporal crop
            start = np.random.randint(0, total_frames - self.num_frames + 1)
            indices = list(range(start, start + self.num_frames))

        # Load and transform frames
        frames = []
        for idx in indices:
            img = Image.open(frame_files[idx])
            img_tensor = self.transform(img)
            frames.append(img_tensor)

        # Stack: [num_frames, C, H, W]
        video = torch.stack(frames, dim=0)

        # Normalize to [-1, 1] (from [0, 1])
        video = video * 2 - 1

        return video

    def __getitem__(self, idx: int) -> torch.Tensor:
        video_dir = self.videos[idx]
        return self._load_frames(video_dir)


def create_ucf101_dataloader(
    root: str = "/datasets01/UCF101_Frames/frames",
    actions: list = None,
    batch_size: int = 16,
    num_frames: int = 16,
    image_size: int = 64,
    train: bool = True,
    grayscale: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create a DataLoader for UCF101.

    Args:
        root: Path to frames directory
        actions: List of action classes (e.g., ["ApplyLipstick"])
        batch_size: Batch size
        num_frames: Frames per clip
        image_size: Target image size
        train: Train or test split
        grayscale: Convert to grayscale
        num_workers: Number of data loading workers

    Returns:
        DataLoader instance
    """
    dataset = UCF101FramesDataset(
        root=root,
        actions=actions,
        num_frames=num_frames,
        image_size=image_size,
        train=train,
        grayscale=grayscale,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
