from __future__ import annotations

from typing import Callable

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_image_transform(dataset_name: str, image_size: int, train: bool) -> Callable:
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    ops = [transforms.Resize((image_size, image_size))]
    if train:
        if dataset_name == "cifar10":
            ops.extend(
                [
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        else:
            ops.append(transforms.RandomHorizontalFlip())

    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(ops)


class HFImageDataset(Dataset):
    def __init__(self, hf_dataset, image_key: str, label_key: str, transform: Callable):
        self.dataset = hf_dataset
        self.image_key = image_key
        self.label_key = label_key
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        image = sample[self.image_key]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        image = self.transform(image)
        label = int(sample[self.label_key])
        return image, label


class HFImageDictDataset(Dataset):
    def __init__(self, hf_dataset, image_key: str, label_key: str, transform: Callable):
        self.dataset = hf_dataset
        self.image_key = image_key
        self.label_key = label_key
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        image = sample[self.image_key]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        pixel_values = self.transform(image)
        label = int(sample[self.label_key])
        return {"pixel_values": pixel_values, "labels": torch.tensor(label, dtype=torch.long)}
