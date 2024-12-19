import segmentation_models_pytorch as smp
import torch

from my_utils.config import DEVICE

class_weights = torch.tensor([0.2, 0.8]).to(DEVICE)

dice_loss = smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE,  smooth=1e-6)
bce_loss = smp.losses.SoftBCEWithLogitsLoss(pos_weight=class_weights[1])
focal_loss = smp.losses.FocalLoss(mode=smp.losses.BINARY_MODE, alpha=0.75, gamma=2.0)


def combined_loss(predicts, targets):
    return 0.7 * bce_loss(predicts, targets) + 0.6 * dice_loss(predicts, targets) + 0.2 * focal_loss(predicts, targets)

