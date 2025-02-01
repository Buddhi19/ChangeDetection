import os
import sys

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dir)

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor, einsum
import torch.nn .functional as F
from CDMamba.misc.torchutils import class2one_hot,simplex
from typing import Iterable, Set, Tuple
from scipy.ndimage import distance_transform_edt

def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def boundary_loss(pred, target):
    target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()  # [B, C, H, W]

    boundary_distances = []
    for b in range(target.shape[0]):  # Iterate over batch
        for c in range(target_onehot.shape[1]):  # Iterate over classes
            mask = target_onehot[b, c].cpu().numpy()  # Get binary mask for class c
            if np.sum(mask) == 0:  # Skip if no pixels for this class
                boundary_distances.append(np.zeros_like(mask))
                continue
            # Compute the distance transform for the foreground (class c)
            pos_dist = distance_transform_edt(mask)
            neg_dist = distance_transform_edt(1 - mask)
            boundary_dist = pos_dist + neg_dist
            boundary_distances.append(boundary_dist)
    
    boundary_distances = np.stack(boundary_distances, axis=0)  # [B*C, H, W]
    boundary_distances = torch.from_numpy(boundary_distances).float().to(pred.device)  # Move to GPU if needed

    boundary_distances = boundary_distances.view(pred.shape)  # [B, C, H, W]

    # Compute the boundary loss
    pred_softmax = F.softmax(pred, dim=1)  # Convert logits to probabilities
    loss = torch.mean(pred_softmax * boundary_distances)  # Weighted sum

    return loss


# def dice_loss(predicts,target,weight=None):
#     idc= [0, 1]
#     probs = torch.softmax(predicts, dim=1)
#     # target = target.unsqueeze(1)
#     target = class2one_hot(target, 7)
#     assert simplex(probs) and simplex(target)

#     pc = probs[:, idc, ...].type(torch.float32)
#     tc = target[:, idc, ...].type(torch.float32)
#     intersection: Tensor = einsum("bcwh,bcwh->bc", pc, tc)
#     union: Tensor = (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))

#     divided: Tensor = torch.ones_like(intersection) - (2 * intersection + 1e-10) / (union + 1e-10)

#     loss = divided.mean()
#     return loss

def dice_loss(pred, target, smooth = 1e-6):
    num_classes = pred.shape[1]
    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()  # (N, C, H, W)
    pred = F.softmax(pred, dim=1)  # (N, C, H, W)
    
    intersection = (pred * target_one_hot).sum(dim=(2, 3))  # (N, C)
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))  # (N, C)
    
    dice = (2. * intersection + smooth) / (union + smooth)  # (N, C)
    
    return 1 - dice.mean()

def ce_dice(input, target, weight=None):
    ce_loss = F.cross_entropy(input, target, ignore_index=255)
    dice_loss_ = dice_loss(input, target)
    loss = 0.5 * ce_loss + 0.5 * dice_loss_
    return loss

def dice(input, target, weight=None):
    dice_loss_ = dice_loss(input, target)
    return dice_loss_

def ce2_dice1(input, target, weight=None):
    ce_loss = F.cross_entropy(input, target, ignore_index=255)
    dice_loss_ = dice_loss(input, target)
    loss = 1.2*ce_loss + 0.75 * dice_loss_ + 0.015 * boundary_loss(input, target)
    return loss

def ce2_dice1_multiclass(input, target, weight=None):
    ce_loss = F.cross_entropy(input, target, ignore_index=255)
    dice_loss_ = dice_loss(input, target)
    loss = ce_loss + 1 * dice_loss_
    return loss


def ce1_dice2(input, target, weight=None):
    ce_loss = F.cross_entropy(input, target, ignore_index=255)
    dice_loss_ = dice_loss(input, target)
    loss = 0.5 * ce_loss +  dice_loss_
    return loss

def ce_scl(input, target, weight=None):
    ce_loss = F.cross_entropy(input, target, ignore_index=255)
    dice_loss_ = dice_loss(input, target)
    loss = 0.5 * ce_loss + 0.5 * dice_loss_
    return loss