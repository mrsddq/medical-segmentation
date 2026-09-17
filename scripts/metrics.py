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


def per_sample_overlap(prediction, target, threshold=0.5):
    """Binary overlap per sample; empty prediction/target pair scores 1."""
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("Prediction and target must have equal batch-first shapes")
    prediction = (prediction > threshold).reshape(prediction.shape[0], -1)
    target = (target > 0.5).reshape(target.shape[0], -1)
    intersection = (prediction & target).sum(1).float()
    total = prediction.sum(1) + target.sum(1)
    union = (prediction | target).sum(1)
    dice = torch.where(total > 0, 2 * intersection / total.clamp(min=1), torch.ones_like(intersection))
    iou = torch.where(union > 0, intersection / union.clamp(min=1), torch.ones_like(intersection))
    return dice, iou
