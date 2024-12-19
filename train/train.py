import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from data.dataloader import create_dataloader
from my_utils.config import *
from my_utils.losses import *
from train_utils import train_one_epoch, validate_one_epoch, test_one_epoch
from models.model import model

loss_fn = focal_loss

train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

train_loader = create_dataloader(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, BATCH_SIZE, train_transform)
val_loader = create_dataloader(VAL_IMAGES_DIR, VAL_MASKS_DIR, BATCH_SIZE, val_transform, shuffle=False)
test_loader = create_dataloader(TEST_IMAGES_DIR, TEST_MASKS_DIR, BATCH_SIZE, val_transform, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
val_losses = []
test_losses = []
train_ious = []
val_ious = []
test_ious = []

best_val_iou = 0.0

resume_training = True

if resume_training:
    try:
        checkpoint = torch.load('best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_iou = checkpoint['best_val_iou']
        print(f"Resuming from epoch {start_epoch} with best_val_iou {best_val_iou:.4f}")
    except FileNotFoundError:
        print("Checkpoint not found. Starting from scratch.")
        start_epoch = 1
else:
    start_epoch = 1

for epoch in range(start_epoch, NUM_EPOCHS + 1):
    print(f"Epoch: {epoch}/{NUM_EPOCHS}")

    train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
    train_losses.append(train_loss)
    train_ious.append(train_iou)
    print(f"Train Loss: {train_loss:.4f}, IoU: {train_iou:.4f}")

    val_loss, val_iou = None, None
    compute_full_metrics = epoch % 10 == 0
    val_loss, val_iou, val_metrics = validate_one_epoch(
        model, val_loader, loss_fn, DEVICE, epoch, compute_full_metrics
    )
    val_losses.append(val_loss)
    val_ious.append(val_iou)
    print(f"Validation Loss: {val_loss:.4f}, IoU: {val_iou:.4f}")

    if compute_full_metrics:
        print(
            f"Validation Dice: {val_metrics['dice']:.4f}, "
            f"Precision: {val_metrics['precision']:.4f}, "
            f"Recall: {val_metrics['recall']:.4f}"
        )

    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'best_val_iou': best_val_iou,
        }, "best_model.pth")
        print(f"Best model saved with IoU: {best_val_iou:.4f}")

test_loss, test_iou, test_metrics = test_one_epoch(model, test_loader, loss_fn, DEVICE, compute_full_metrics=True)
test_losses.append(test_loss)
test_ious.append(test_iou)
print(f"Test Loss: {test_loss:.4f}, IoU: {test_iou:.4f}")
print(
    f"Test Dice: {test_metrics['dice']:.4f}, "
    f"Precision: {test_metrics['precision']:.4f}, "
    f"Recall: {test_metrics['recall']:.4f}"
)

epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses[:len(epochs)], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_ious, label="Train IoU")
plt.plot(epochs, val_ious[:len(epochs)], label="Validation IoU")
plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.title("Training and Validation IoU")
plt.legend()

plt.tight_layout()
plt.show()
