"""
Transform pipelines for ImageNet-pretrained backbones (ResNet18 etc.).
Uses ImageNet normalization stats, NOT the EuroSAT-specific stats from
Phase 6 — pretrained weights expect the same normalization distribution
they were trained on.
"""

import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms_pretrained() -> T.Compose:
    return T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomChoice([
            T.RandomRotation(degrees=(0, 0)),
            T.RandomRotation(degrees=(90, 90)),
            T.RandomRotation(degrees=(180, 180)),
            T.RandomRotation(degrees=(270, 270)),
        ]),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms_pretrained() -> T.Compose:
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])