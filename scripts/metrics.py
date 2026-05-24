import torch
import torch.nn.functional as F


def dice_score(prediction, target, threshold=0.5, smooth=1.0):
    prediction = (prediction > threshold).float().reshape(-1)
    target = target.float().reshape(-1)
    intersection = (prediction * target).sum()
    return (2.0 * intersection + smooth) / (prediction.sum() + target.sum() + smooth)


def dice_loss(prediction, target, smooth=1.0):
    prediction = prediction.reshape(-1)
    target = target.float().reshape(-1)
    intersection = (prediction * target).sum()
    return 1.0 - (2.0 * intersection + smooth) / (prediction.sum() + target.sum() + smooth)


def dice_bce_loss(prediction, target, dice_weight=0.5, bce_weight=0.5):
    return (
        bce_weight * F.binary_cross_entropy(prediction, target.float())
        + dice_weight * dice_loss(prediction, target)
    )
