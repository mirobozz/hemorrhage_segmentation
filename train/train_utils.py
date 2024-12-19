import torch
from tqdm import tqdm
from my_utils.metrics import compute_iou, compute_dice_sim, compute_precision, compute_recall


def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    running_loss, running_iou = 0.0, 0.0

    for images, masks in tqdm(train_loader):
        images, masks = images.to(device), masks.to(device).float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        iou = compute_iou(outputs, masks)
        running_iou += iou

    avg_loss = running_loss / len(train_loader)
    avg_iou = running_iou / len(train_loader)
    return avg_loss, avg_iou


def validate_one_epoch(model, val_loader, loss_fn, device, epoch, compute_full_metrics=False):
    model.eval()
    val_loss, val_iou = 0.0, 0.0
    metrics = {'dice': 0.0, 'precision': 0.0, 'recall': 0.0}

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device).float()
            outputs = model(images)
            loss = loss_fn(outputs, masks.float())
            val_loss += loss.item()

            iou = compute_iou(outputs, masks)
            val_iou += iou

            # Compute additional metrics every 5 epochs
            if compute_full_metrics:
                metrics['dice'] += compute_dice_sim(outputs, masks)
                metrics['precision'] += compute_precision(outputs, masks)
                metrics['recall'] += compute_recall(outputs, masks)

    avg_loss = val_loss / len(val_loader)
    avg_iou = val_iou / len(val_loader)

    if compute_full_metrics:
        metrics = {k: v / len(val_loader) for k, v in metrics.items()}

    return avg_loss, avg_iou, metrics


def test_one_epoch(model, test_loader, loss_fn, device, compute_full_metrics=False):
    model.eval()
    test_loss, test_iou = 0.0, 0.0
    metrics = {'dice': 0.0, 'precision': 0.0, 'recall': 0.0}

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device).float()
            outputs = model(images)
            loss = loss_fn(outputs, masks.float())
            test_loss += loss.item()

            iou = compute_iou(outputs, masks)
            test_iou += iou

            # Compute additional metrics every 5 epochs
            if compute_full_metrics:
                metrics['dice'] += compute_dice_sim(outputs, masks)
                metrics['precision'] += compute_precision(outputs, masks)
                metrics['recall'] += compute_recall(outputs, masks)

    avg_test_loss = test_loss / len(test_loader)
    avg_test_iou = test_iou / len(test_loader)

    if compute_full_metrics:
        metrics = {k: v / len(test_loader) for k, v in metrics.items()}

    return avg_test_loss, avg_test_iou, metrics
