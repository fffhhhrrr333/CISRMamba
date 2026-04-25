import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import glob
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.image as mpimg
import albumentations as A
from torch import nn
from config import *
from util import *
from sklearn.model_selection import train_test_split
import torch.optim.lr_scheduler as lr_scheduler
import shutil
from dataloaded import *
from CISRMamba import *
import pandas as pd

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

if torch.cuda.is_available():
    print(f"Using Device: {DEVICE}")
else:
    print("Using CPU (Warning: Training might be slow)")

model = CISRMamba(num_classes=1)

if loadstate and os.path.exists(loadstateptfile):
    print(f"Loading checkpoint from {loadstateptfile}...")
    model.load_state_dict(torch.load(loadstateptfile))
else:
    print("Initializing model from scratch...")

model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
print(f"Scheduler: CosineAnnealingLR (T_max={EPOCHS}, eta_min=1e-6)")

TEST_OPT_DIR = "/test/opt"
TEST_SAR_DIR = "/test/vv"
TEST_LBL_DIR = "/test/flood_vv"

print(f"Loading Test Dataset from: {TEST_OPT_DIR}")

test_transform = A.Compose([
    A.ToFloat(255),
    A.Resize(height, width), 
    ToTensorV2(),
])

try:
    test_dataset = CustomDataset(
        optical_dir=TEST_OPT_DIR,
        radar_dir=TEST_SAR_DIR,
        label_dir=TEST_LBL_DIR,
        transform=test_transform
    )
    
    # Batch Size = 1
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f"✅ Test Loader Ready. Found {len(test_dataset)} samples.")

except Exception as e:
    print(f"\n❌ Error")
    print(f"Detail: {e}\n")
    exit()

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

outloss = {
    'epoch': [],
    'train_loss': [],
    'test_loss': [],
    'mIoU': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1': [],
    'Kappa': []
}

best_miou = 0.0 
best_epoch = 0


print(f"Start Training for {EPOCHS} Epochs...")

for i in range(EPOCHS):
    epoch_idx = i + 1
    model.train()

    train_loss = train_fn(train_loader, model, optimizer)

    model.eval()
    test_loss, test_metrics = eval_fn(test_loader, model, outfile=None)

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    if test_metrics is None:
        test_metrics = {k: 0.0 for k in ['mIoU', 'Accuracy', 'Precision', 'Recall', 'F1', 'Kappa']}

    curr_miou = test_metrics.get('mIoU', 0.0)
    curr_f1 = test_metrics.get('F1', 0.0)

    outloss['epoch'].append(epoch_idx)
    outloss['train_loss'].append(train_loss)
    outloss['test_loss'].append(test_loss)
    outloss['mIoU'].append(curr_miou)
    outloss['Accuracy'].append(test_metrics.get('Accuracy', 0.0))
    outloss['Precision'].append(test_metrics.get('Precision', 0.0))
    outloss['Recall'].append(test_metrics.get('Recall', 0.0))
    outloss['F1'].append(curr_f1)
    outloss['Kappa'].append(test_metrics.get('Kappa', 0.0))

    print(f"\nEpoch: {epoch_idx}/{EPOCHS} | LR: {current_lr:.6f}")
    print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
    print(f"Test Metrics -> mIoU: {curr_miou:.2f}% | F1: {curr_f1:.2f}%")

    last_ckpt_path = os.path.join(CHECKPOINT_DIR, "last_model.pt")
    torch.save(model.state_dict(), last_ckpt_path)

    if curr_miou > best_miou:
        previous = best_miou
        best_miou = curr_miou
        best_epoch = epoch_idx

        best_ckpt_path = os.path.join(CHECKPOINT_DIR, f"{ENCODER}_{WEIGHTS}_{name}_best.pt")
        torch.save(model.state_dict(), best_ckpt_path)
        print(f"🔥 New Best Test mIoU! ({previous:.2f}% -> {best_miou:.2f}%) Model Saved.")
    else:
        print(f"Current Test mIoU: {curr_miou:.2f}% (Best: {best_miou:.2f}% at Epoch {best_epoch})")

    outcsv = pd.DataFrame(outloss)
    outcsv.to_csv(METRICS_CSV, index=False)

print("\n------------------------------------------------")
print(f"Training Complete! Final Best Test mIoU: {best_miou:.2f}% at Epoch {best_epoch}")
print("------------------------------------------------")