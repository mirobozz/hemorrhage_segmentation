import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.utils.metrics import Fscore, Precision, Recall, Accuracy


def compute_iou(outputs, masks, threshold=0.5):
    outputs = outputs.float()
    masks = masks.long()

    if masks.max() > 1:
        masks = masks / 255

    outputs = torch.sigmoid(outputs) if outputs.dtype != torch.float32 else outputs

    assert outputs.shape == masks.shape, f"Shape mismatch: {outputs.shape} vs {masks.shape}"

    tp, fp, fn, tn = smp.metrics.get_stats(outputs, masks, mode='binary', threshold=threshold)

    iou_score = torch.where(tp + fp + fn > 0, tp / (tp + fp + fn), torch.tensor(0.0, device=tp.device))

    return iou_score.mean().item()



def compute_dice_sim(outputs, masks, threshold=0.5):
    dice_coefficient = Fscore(threshold=threshold)
    dice_score = dice_coefficient(outputs, masks)
    return dice_score


def compute_precision(outputs, masks, threshold=0.5):
    precision_score = Precision(threshold=threshold)
    precision = precision_score(outputs, masks)
    return precision.item()


def compute_recall(outputs, masks, threshold=0.5):
    recall_score = Recall(threshold=threshold)
    recall = recall_score(outputs, masks)
    return recall.item()



