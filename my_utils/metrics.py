import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.utils.metrics import Fscore, Precision, Recall


def binarize(outputs, masks, threshold=0.5):
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > threshold).int()
    masks = (masks > 0.5).int()
    return outputs, masks

def compute_iou(outputs, masks, threshold=0.5):
    outputs, masks = binarize(outputs, masks, threshold)
    tp, fp, fn, tn = smp.metrics.get_stats(outputs, masks, mode='binary', threshold=None)
    iou_score = torch.where(tp + fp + fn > 0, tp / (tp + fp + fn), torch.tensor(0.0, device=tp.device))
    return iou_score.mean().item()


def compute_dice_sim(outputs, masks, threshold=0.5):
    outputs, masks = binarize(outputs, masks, threshold)
    return Fscore(threshold=None)(outputs, masks).item()


def compute_precision(outputs, masks, threshold=0.5):
    outputs, masks = binarize(outputs, masks, threshold)
    return Precision(threshold=None)(outputs, masks).item()


def compute_recall(outputs, masks, threshold=0.5):
    outputs, masks = binarize(outputs, masks, threshold)
    return Recall(threshold=None)(outputs, masks).item()

def compute_f1(outputs, masks, threshold=0.5):
    outputs, masks = binarize(outputs, masks, threshold)
    f1_metric = Fscore(threshold=None)
    return f1_metric(outputs, masks).item()

