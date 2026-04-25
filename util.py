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
import numpy as np
import matplotlib.pyplot as plt

def linear_stretch(image, percent):
    stretched_image = np.zeros_like(image, dtype=np.float32)
    
    for i in range(image.shape[0]):
        band = image[i, :, :]
        
        flat_band = band.flatten()
        
        low_percent = np.percentile(flat_band, percent)
        high_percent = np.percentile(flat_band, 100 - percent)
        
        stretched_band = np.clip((band - low_percent) / (high_percent - low_percent + 1e-8), 0, 1)
        
        stretched_image[i, :, :] = stretched_band
    
    return stretched_image

def train_fn(data_loader, model, optimizer):
    model.train()
    total_lossn = 0
    train_bar = tqdm(data_loader)
    for images, masks in train_bar:
        images = images.to(DEVICE, dtype=torch.float32)
        masks = masks.to(DEVICE, dtype=torch.float32)

        optimizer.zero_grad()

        logits, diceloss, bceloss = model(images, masks)
        
        total_loss = diceloss + bceloss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_lossn += total_loss.item()
        
        train_bar.set_description("Train loss: %.4f" % (
            total_lossn / (train_bar.n + 1), 
        ))
    return total_lossn / len(data_loader)

def calculate_acc(predictions, masks):
    pred_ones = predictions == 1
    mask_ones = masks == 1
    
    correct_ones = torch.logical_and(pred_ones, masks == 1).sum().float().item()

    num_pred_ones = mask_ones.sum().float().item()
    
    if num_pred_ones == 0:
        return 0.0
    
    acc = correct_ones / num_pred_ones
    return acc

def eval_fn(data_loader, model, outfile):
    model.eval()
    test_bar = tqdm(data_loader)
    total_lossn = 0

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    with torch.no_grad():
        for images, masks in test_bar:
            images = images.to(DEVICE, dtype=torch.float32)
            masks = masks.to(DEVICE, dtype=torch.float32)

            logits, diceloss, bceloss = model(images, masks)
            total_loss = diceloss + bceloss
            total_lossn += total_loss.item()

            pred_mask = torch.sigmoid(logits)
            predict = (pred_mask > ratio).long()
            gt = masks.long().to(DEVICE)

            p = predict.view(-1)
            g = gt.view(-1)

            TP += int(((p == 1) & (g == 1)).sum().item())
            TN += int(((p == 0) & (g == 0)).sum().item())
            FP += int(((p == 1) & (g == 0)).sum().item())
            FN += int(((p == 0) & (g == 1)).sum().item())

            totals = TP + TN + FP + FN + 1e-12
            running_acc = (TP + TN) / totals
            test_bar.set_description("Test ACC: %.4f" % (running_acc,))

        # compute metrics from confusion matrix
        eps = 1e-7
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
        iou_pos = TP / (TP + FP + FN + eps)
        iou_neg = TN / (TN + FN + FP + eps)
        miou = (iou_pos + iou_neg) / 2.0

        N = TP + TN + FP + FN + eps
        po = accuracy
        pe = (((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN))) / (N * N)
        kappa = (po - pe) / (1 - pe + eps)

        metrics = {
            'mIoU': miou * 100.0,
            'Accuracy': accuracy * 100.0,
            'Precision': precision * 100.0,
            'Recall': recall * 100.0,
            'F1': f1 * 100.0,
            'Kappa': kappa
        }

        # Visualization
        if outfile is not None:

            vis_images, vis_masks = next(iter(data_loader))  # noqa: use first batch for viz
            current_batch_size = vis_images.size(0)

            sample_idx = np.random.randint(0, current_batch_size)
            
            image = vis_images[sample_idx]
            mask = vis_masks[sample_idx]
            
            img_tensor = image.to(DEVICE, dtype=torch.float32).unsqueeze(0)
            

            try:
                res = model(img_tensor)
                if isinstance(res, tuple):
                    logits_mask = res[0]
                else:
                    logits_mask = res
            except:
                dummy_mask = torch.zeros_like(img_tensor[:, 0:1, :, :])
                res = model(img_tensor, dummy_mask)
                if isinstance(res, tuple):
                    logits_mask = res[0]
                else:
                    logits_mask = res

            pred_mask = torch.sigmoid(logits_mask)
            pred_mask = (pred_mask > ratio) * 1.0


            if image.shape[0] >= 3:
                vis_img = linear_stretch(image.numpy()[[2, 1, 0]], 2)
            else:

                vis_img = linear_stretch(image.numpy(), 2)
                
            vis_img = np.uint8(vis_img * 255)

            f, axarr = plt.subplots(1, 3, figsize=(12, 4))
            axarr[1].imshow(np.squeeze(mask.numpy()), cmap='gray', vmin=0, vmax=1)
            axarr[1].set_title('Ground Truth')
            
            if len(vis_img.shape) == 3:
                axarr[0].imshow(np.transpose(vis_img, (1, 2, 0)))
            else:
                axarr[0].imshow(vis_img, cmap='gray')
            axarr[0].set_title('Input Image')
            
            axarr[2].imshow(pred_mask.detach().cpu().squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
            axarr[2].set_title('Prediction')
            
            plt.tight_layout()
            plt.savefig(outfile, pad_inches=0.01, bbox_inches='tight')
            plt.close()

    avg_loss = total_lossn / len(data_loader)
    return avg_loss, metrics