import torch
from tqdm import tqdm
from segmentation_models_pytorch.utils.metrics import IoU, Fscore, Precision, Recall


def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    running_loss, running_iou = 0.0, 0.0
    iou_metric = IoU(threshold=0.5).to(device)

    for images, masks in tqdm(train_loader):
        images, masks = images.to(device), masks.to(device).float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_iou += iou_metric(outputs, masks).item()

    avg_loss = running_loss / len(train_loader)
    avg_iou = running_iou / len(train_loader)
    return avg_loss, avg_iou


def validate_one_epoch(model, val_loader, loss_fn, device, compute_full_metrics=False):
    model.eval()
    val_loss, val_iou = 0.0, 0.0
    iou_metric = IoU(threshold=0.5).to(device)

    if compute_full_metrics:
        dice_metric = Fscore(threshold=0.5).to(device)
        precision_metric = Precision(threshold=0.5).to(device)
        recall_metric = Recall(threshold=0.5).to(device)
        metrics = {'dice': 0.0, 'precision': 0.0, 'recall': 0.0}

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device).float()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()
            val_iou += iou_metric(outputs, masks).item()

            if compute_full_metrics:
                metrics['dice'] += dice_metric(outputs, masks).item()
                metrics['precision'] += precision_metric(outputs, masks).item()
                metrics['recall'] += recall_metric(outputs, masks).item()

    avg_loss = val_loss / len(val_loader)
    avg_iou = val_iou / len(val_loader)

    if compute_full_metrics:
        metrics = {k: v / len(val_loader) for k, v in metrics.items()}
        return avg_loss, avg_iou, metrics

    return avg_loss, avg_iou, {}


def test_one_epoch(model, test_loader, loss_fn, device, compute_full_metrics=False):
    model.eval()
    test_loss, test_iou = 0.0, 0.0
    iou_metric = IoU(threshold=0.5).to(device)

    if compute_full_metrics:
        dice_metric = Fscore(threshold=0.5).to(device)
        precision_metric = Precision(threshold=0.5).to(device)
        recall_metric = Recall(threshold=0.5).to(device)
        metrics = {'dice': 0.0, 'precision': 0.0, 'recall': 0.0}

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device).float()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            test_loss += loss.item()
            test_iou += iou_metric(outputs, masks).item()

            if compute_full_metrics:
                metrics['dice'] += dice_metric(outputs, masks).item()
                metrics['precision'] += precision_metric(outputs, masks).item()
                metrics['recall'] += recall_metric(outputs, masks).item()

    avg_loss = test_loss / len(test_loader)
    avg_iou = test_iou / len(test_loader)

    if compute_full_metrics:
        metrics = {k: v / len(test_loader) for k, v in metrics.items()}
        return avg_loss, avg_iou, metrics

    return avg_loss, avg_iou, {}
