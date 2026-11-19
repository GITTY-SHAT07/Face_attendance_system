
import os
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ORLDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    return train_transform, val_transform


def load_orl_samples(root):
    samples = []
    class_to_idx = {}

    subject_folders = sorted(
        folder for folder in os.listdir(root)
        if os.path.isdir(os.path.join(root, folder))
    )

    for label, subject in enumerate(subject_folders):
        subject_path = os.path.join(root, subject)
        class_to_idx[subject] = label

        image_files = sorted(
            file for file in os.listdir(subject_path)
            if file.lower().endswith((".pgm", ".png", ".jpg", ".jpeg"))
        )

        for image_file in image_files:
            image_path = os.path.join(subject_path, image_file)
            samples.append((image_path, label))

    if not samples:
        raise RuntimeError(f"No images found in {root}")

    return samples, class_to_idx


def split_samples(samples, val_ratio=0.2, seed=42):
    random.seed(seed)

    samples_by_identity = defaultdict(list)

    for image_path, label in samples:
        samples_by_identity[label].append((image_path, label))

    train_samples = []
    val_samples = []

    for identity_samples in samples_by_identity.values():
        random.shuffle(identity_samples)

        val_count = max(1, round(len(identity_samples) * val_ratio))

        val_samples.extend(identity_samples[:val_count])
        train_samples.extend(identity_samples[val_count:])

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    return train_samples, val_samples


def create_dataloaders(
    root="training_data/att_faces",
    batch_size=16,
    val_ratio=0.2
):
    samples, class_to_idx = load_orl_samples(root)

    train_samples, val_samples = split_samples(
        samples,
        val_ratio
    )

    train_transform, val_transform = get_transforms()

    train_dataset = ORLDataset(
        train_samples,
        train_transform
    )

    val_dataset = ORLDataset(
        val_samples,
        val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, class_to_idx


if __name__ == "__main__":
    train_loader, val_loader, class_to_idx = create_dataloaders()

    print("Dataset loaded successfully")
    print("Number of identities:", len(class_to_idx))
    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))

    images, labels = next(iter(train_loader))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Unique identities in batch:", labels.unique().tolist())
