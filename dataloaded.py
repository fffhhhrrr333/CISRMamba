import os
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
import albumentations as A
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from config import *

class CustomDataset(Dataset):
    def __init__(self, optical_dir, radar_dir, label_dir, transform=None):
        self.optical_dir = optical_dir
        self.radar_dir = radar_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_names = os.listdir(optical_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]

        # Load optical image (4 channels)
        optical_path = os.path.join(self.optical_dir, image_name)
        optical_image = Image.open(optical_path)
        
        # Load radar image (1 channel)
        radar_path = os.path.join(self.radar_dir, image_name)
        radar_image = Image.open(radar_path)
        
        # Load label image (binary mask)
        label_path = os.path.join(self.label_dir, image_name)
        label_image = np.array(Image.open(label_path))

        # Stack optical and radar images
        optical_image = np.array(optical_image)
        radar_image = np.expand_dims(np.array(radar_image), axis=-1)  # Reshape to (H, W, 1)
        combined_image = np.concatenate((optical_image, radar_image), axis=-1)  # Shape (H, W, 5)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=combined_image, mask=label_image)
            combined_image = augmented['image']
            label_image = augmented['mask']
            return combined_image, label_image.unsqueeze(0).float()

        # transform=None: return raw numpy for SubsetWithTransform to handle
        return combined_image, label_image

optical_dir = OPTICAL_DIR
radar_dir = RADAR_DIR
label_dir = LABEL_DIR

train_transform = A.Compose([
    A.ToFloat(255),
    A.Resize(height, width),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.ToFloat(255),
    A.Resize(height, width),
    ToTensorV2(),
])

# Create dataset (no augmentation at split time; apply per-subset below)
base_dataset = CustomDataset(optical_dir, radar_dir, label_dir, transform=None)

# Split dataset into training and testing sets
train_size = int(0.8 * len(base_dataset))
val_size = len(base_dataset) - train_size
train_indices, val_indices = torch.utils.data.random_split(
    range(len(base_dataset)), [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

class SubsetWithTransform(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
        self.dataset.transform = None  # ensure base has no transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, mask = self.dataset[self.indices[idx]]
        if self.transform:
            mask_np = mask.squeeze(0).numpy() if hasattr(mask, 'numpy') else mask
            img_np = img if not hasattr(img, 'numpy') else img
            augmented = self.transform(image=img_np, mask=mask_np)
            img = augmented['image']
            mask = augmented['mask'].unsqueeze(0).float()
        return img, mask

train_dataset = SubsetWithTransform(base_dataset, train_indices, train_transform)
val_dataset   = SubsetWithTransform(base_dataset, val_indices,   val_transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
valid_loader = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

