from typing import Optional

import cv2
import torch
import torchvision
from groundingdino.util.inference import annotate, batch_predict, load_image, load_model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose(
    [
        transforms.Resize([800, 1000]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def create_dataloader(
    data_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[torchvision.transforms.Compose] = None,
) -> DataLoader:
    """
    Creates a PyTorch DataLoader for images stored in subdirectories.

    Parameters:
    - data_dir (str): Path to the main directory containing subdirectories of images.
    - batch_size (int): Number of samples per batch to load.
    - shuffle (bool): Whether to shuffle the dataset.
    - num_workers (int): How many subprocesses to use for data loading.
    - transform (torchvision.transforms.Compose): Transformations to apply to the images.

    Returns:
    - DataLoader: PyTorch DataLoader.
    """

    # Load the dataset from the directory with subdirectories
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Create the DataLoader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return dataloader


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_data_dir", type=str, default="data")