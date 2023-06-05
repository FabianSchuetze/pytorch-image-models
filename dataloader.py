import math

import torch
from torchvision import datasets
from datasets import load_dataset
from datasets.utils.info_utils import VerificationMode
import torchvision.transforms as transforms
from PIL import Image
from typing import Optional, Callable, Any, Tuple

def load_data(example):
    image = example['image'][0]
    example["image"] = [image.convert("RGB")]
    return example

ds = load_dataset('/workspace/imagenet-1k/', split='validation', verification_mode=VerificationMode.NO_CHECKS)
ds.set_transform(load_data)


class DatasetFolder(torch.utils.data.Dataset):
    """A generic data loader.

    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__( self, huggingsface_ds, transform) -> None:
        super().__init__()
        self.transform =transform
        self.ds = huggingsface_ds

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        example = self.ds[index]
        sample, target = example['image'], example['label']
        # path, target = self.samples[index]
        # sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        # if self.target_transform is not None:
            # target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.ds)


def build_transform(input_size=224,
                    interpolation='bicubic',
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    crop_pct=0.875):

    def _pil_interp(method):
        if method == 'bicubic':
            return Image.BICUBIC
        elif method == 'lanczos':
            return Image.LANCZOS
        elif method == 'hamming':
            return Image.HAMMING
        else:
            return Image.BILINEAR

    resize_im = input_size > 32
    t = []
    if resize_im:
        size = int(math.floor(input_size / crop_pct))
        ip = _pil_interp(interpolation)
        t.append(
            transforms.Resize(
                size,
                interpolation=ip),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
crop_pct = 0.9

def load_val_dataset():
    train_transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)
    test = DatasetFolder(ds, train_transform)
    return test

