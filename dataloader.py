import math
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Optional, Callable, Any, Tuple

from classes import IMAGENET2012_CLASSES

# def load_data(example):
    # image = example['image'][0]
    # example["image"] = [image.convert("RGB")]
    # return example

# ds = load_dataset('/workspace/imagenet-1k/', split='validation', verification_mode=VerificationMode.NO_CHECKS)


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

    def __init__(self, folder, transform) -> None:
        super().__init__()
        self.files = [os.path.join(folder, i) for i in os.listdir(folder)
                      if '.JPEG' in i]
        self.labels = IMAGENET2012_CLASSES.values()
        self.label_idx = {i: idx for idx, i in enumerate(self.labels)}
        self.transform = transform

    def load_label(self, path):
        root, _ = os.path.splitext(path)
        _, synset_id = os.path.basename(root).rsplit("_", 1)
        label = IMAGENET2012_CLASSES[synset_id]
        return label

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        file = self.files[index]
        target = self.load_label(file)
        label = self.label_idx[target]
        sample = Image.open(file)
        sample = sample.convert("RGB")
        # example = self.ds[index]
        # sample, target = example['image'], example['label']
        # path, target = self.samples[index]
        # sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        # if self.target_transform is not None:
            # target = self.target_transform(target)

        return sample, label

    def __len__(self) -> int:
        return len(self.files)


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
    test = DatasetFolder('/workspace/imagenet/', train_transform)
    loader = torch.utils.data.DataLoader(
            test,
            batch_size=100,
            shuffle=False,
            num_workers=4)
    return loader

