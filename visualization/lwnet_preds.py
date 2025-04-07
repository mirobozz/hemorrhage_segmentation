import cv2
import numpy as np
import torch
import os
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import data.dataloader as dl
import my_utils.config
import albumentations as A
import pandas as pd

colors = {
    "TP": [0, 255, 0],
    "FP": [0, 0, 255],
    "TN": [255, 0, 0],
    "FN": [255, 193, 203],
}


def compute_iou(pred, mask):
    pred = pred.astype(bool)
    mask = mask.astype(bool)
    intersection = np.logical_and(pred, mask).sum()
    union = np.logical_or(pred, mask).sum()
    if union == 0:
        return float('1.0')
    return intersection / union


def save_predictions(model, dataloader, output_dir, alpha=0.5):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    ious = []

    with torch.no_grad():
        for idx, (images, masks) in enumerate(dataloader):
            images = images.to(my_utils.config.DEVICE)
            masks = masks.to(my_utils.config.DEVICE)

            predictions = torch.sigmoid(model(images))

            for i in range(images.shape[0]):
                image = images[i].permute(1, 2, 0).cpu().numpy()
                image = (image * std + mean)
                image = (image * 255).clip(0, 255).astype(np.uint8)

                mask = masks[i].squeeze(0).cpu().numpy()
                mask = (mask > 0.5).astype(np.uint8) * 255

                prediction = predictions[i].squeeze(0).cpu().numpy()
                prediction = np.where(prediction > 0.55, 1, 0).astype(np.uint8) * 255

                iou = compute_iou(prediction > 0, mask > 0)
                ious.append({
                    "image_name": f"result_{idx}_{i}.png",
                    "IoU": iou
                })

                tp = (mask == 255) & (prediction == 255)
                fp = (mask == 0) & (prediction == 255)
                tn = (mask == 0) & (prediction == 0)
                fn = (mask == 255) & (prediction == 0)

                overlay = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
                overlay[tp] = colors["TP"]
                overlay[fp] = colors["FP"]
                overlay[tn] = colors["TN"]
                overlay[fn] = colors["FN"]

                combined_overlay = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

                combined_image = np.hstack((
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR),
                    combined_overlay
                ))

                cv2.imwrite(os.path.join(output_dir, f"result_{idx}_{i}.png"), combined_image)

    iou_df = pd.DataFrame(ious)
    iou_df.to_csv(os.path.join(output_dir, "iou_metrics.csv"), index=False)


if __name__ == "__main__":
    images_dir = my_utils.config.TEST_IMAGES_DIR
    masks_dir = my_utils.config.TEST_MASKS_DIR
    output_dir = "C:/Users/mirom/Desktop/preds_temp"

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    batch_size = my_utils.config.BATCH_SIZE
    dataloader = dl.create_dataloader(
        images_dir=images_dir,
        masks_dir=masks_dir,
        batch_size=batch_size,
        transform=transform,
        shuffle=False,
    )

    model = smp.Unet(
        encoder_name=my_utils.config.ENCODER,
        encoder_weights=None,
        classes=len(my_utils.config.CLASSES),
        activation=my_utils.config.ACTIVATION,
    )

    model_path = "../train/lwnet_trained.pth"
    #checkpoint = torch.load(model_path, map_location=my_utils.config.DEVICE)
    #model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(my_utils.config.DEVICE)

    save_predictions(model, dataloader, output_dir)
