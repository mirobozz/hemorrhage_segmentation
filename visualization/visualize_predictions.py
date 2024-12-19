# TODO plt(RGB) -> cv2(BGR) +


import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

import data.dataloader as dl
import my_utils.config
import matplotlib.pyplot as plt
import albumentations as A

colors = {
    "TP": [0, 255, 0],
    "FP": [0, 0, 255],
    "TN": [255, 0, 0],
    "FN": [255, 193, 203],
}


def visualize_predictions(model, dataloader, alpha=0.5):
    model.eval()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for images, masks in dataloader:

            images = images.to(my_utils.config.DEVICE)
            masks = masks.to(my_utils.config.DEVICE)

            predictions = torch.sigmoid(model(images))

            for i in range(images.shape[0]):

                image = images[i].permute(1, 2, 0).cpu().numpy()
                image = (image * std + mean)
                image = (image * 255).clip(0, 255).astype(np.uint8)

                mask = masks[i].squeeze(0).cpu().numpy()
                mask = (mask > 0.5).astype(np.uint8)

                prediction = predictions[i].squeeze(0).cpu().numpy()
                prediction = np.where(prediction > 0.55, 1, 0).astype(np.uint8)

                if np.array_equal(mask, prediction):
                    print("Warning: Ground Truth and prediction are identical!")

                intersection = np.logical_and(mask, prediction).sum()
                union = np.logical_or(mask, prediction).sum()
                iou = intersection / union if union > 0 else 0
                dice = (2 * intersection) / (mask.sum() + prediction.sum()) if (mask.sum() + prediction.sum()) > 0 else 0

                print(f"IoU: {iou:.4f}, Dice Score: {dice:.4f}")

                tp = (mask == 1) & (prediction == 1)
                fp = (mask == 0) & (prediction == 1)
                tn = (mask == 0) & (prediction == 0)
                fn = (mask == 1) & (prediction == 0)

                overlay = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
                overlay[tp] = colors["TP"]
                overlay[fp] = colors["FP"]
                overlay[tn] = colors["TN"]
                overlay[fn] = colors["FN"]

                combined_overlay = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

                plt.figure(figsize=(20, 5))
                plt.subplot(1, 4, 1)
                plt.title("Original Image")
                plt.imshow(image)
                plt.axis("off")

                plt.subplot(1, 4, 2)
                plt.title("Ground Truth")
                plt.imshow(mask, cmap='gray')
                plt.axis("off")

                plt.subplot(1, 4, 3)
                plt.title("Prediction")
                plt.imshow(prediction * 255, cmap='gray')
                plt.axis("off")

                plt.subplot(1, 4, 4)
                plt.title("Prediction Visualization")
                plt.imshow(cv2.cvtColor(combined_overlay, cv2.COLOR_BGR2RGB))
                plt.axis("off")

                plt.show()


if __name__ == "__main__":
    images_dir = my_utils.config.TEST_IMAGES_DIR
    masks_dir = my_utils.config.TEST_MASKS_DIR

    transform = A.Compose([
        A.Resize(128, 128),
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

    model_path = "../train/best_model.pth"

    checkpoint = torch.load(model_path, map_location=my_utils.config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(my_utils.config.DEVICE)

    visualize_predictions(model, dataloader)
