import os
import cv2
import numpy as np
import torch
import pandas as pd
from albumentations.pytorch import ToTensorV2
import albumentations as A
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.metrics import get_stats
from segmentation_models_pytorch.utils.metrics import Precision, Recall
import my_utils.config
from my_utils.config import TEST_IMAGES_DIR

PATCH_SIZE = 256
SHIFT_X, SHIFT_Y = PATCH_SIZE // 2, PATCH_SIZE // 2

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
        return float("1.0")
    return intersection / union


def crop_bb_by_image(image, mask):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    non_zero_mask = grayscale > 0
    coords = np.argwhere(non_zero_mask)
    if coords.size == 0:
        return image, mask
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return image[y_min : y_max + 1, x_min : x_max + 1], mask[y_min : y_max + 1, x_min : x_max + 1]


def resize_to_nearest(image: np.ndarray, base: int = PATCH_SIZE) -> np.ndarray:
    h, w = image.shape[:2]
    new_h = (h // base) * base
    new_w = (w // base) * base
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def split_into_patches_with_coords(image: np.ndarray, stride: int) -> list:
    patches = []
    h, w = image.shape[:2]
    for y in range(0, h - PATCH_SIZE + 1, stride):
        for x in range(0, w - PATCH_SIZE + 1, stride):
            patch = image[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
            patches.append((patch, y, x))
    return patches


def save_predictions_on_patches(model, images_dir, masks_dir, output_dir, transform, alpha=0.5, threshold=0.7):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    metrics = []

    image_names = sorted(os.listdir(images_dir))
    mask_names = sorted(os.listdir(masks_dir))

    with torch.no_grad():
        for idx, (img_name, mask_name) in enumerate(zip(image_names, mask_names)):
            img_path = os.path.join(images_dir, img_name)
            mask_path = os.path.join(masks_dir, mask_name)

            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                continue

            image, mask = crop_bb_by_image(image, mask)
            image = resize_to_nearest(image, base=PATCH_SIZE)
            mask = resize_to_nearest(mask, base=PATCH_SIZE)

            h, w = image.shape[:2]
            vote_sum = np.zeros((h, w), dtype=np.float32)
            vote_count = np.zeros((h, w), dtype=np.uint8)

            for stride in [int(PATCH_SIZE*0.5)]:
                for y in range(0, h - PATCH_SIZE + 1, stride):
                    for x in range(0, w - PATCH_SIZE + 1, stride):
                        patch = image[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                        aug = transform(image=patch)
                        input_tensor = aug["image"].unsqueeze(0).to(my_utils.config.DEVICE)
                        pred = torch.sigmoid(model(input_tensor))[0, 0].cpu().numpy()
                        vote_sum[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += pred
                        vote_count[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += 1

            vote_count[vote_count == 0] = 1
            final_mask = (vote_sum / vote_count) > threshold
            full_prediction = final_mask.astype(np.uint8) * 255

            iou = compute_iou(full_prediction > 0, mask > 0)

            pred_tensor = torch.tensor((full_prediction > 0).astype(np.uint8)).unsqueeze(0).unsqueeze(0)
            mask_tensor = torch.tensor((mask > 0).astype(np.uint8)).unsqueeze(0).unsqueeze(0)

            tp, fp, fn, tn = get_stats(pred_tensor, mask_tensor, mode='binary')

            precision = Precision(threshold=None)(pred_tensor, mask_tensor).item()
            recall = Recall(threshold=None)(pred_tensor, mask_tensor).item()
            accuracy = torch.where(tp + fp + fn + tn > 0, (tp + tn) / (tp + fp + fn + tn), torch.tensor(0.0)).mean().item()

            metrics.append({
                "image_name": img_name,
                "IoU": iou,
                "Precision": precision,
                "Recall": recall,
                "Accuracy": accuracy
            })

            tp_mask = (mask == 255) & (full_prediction == 255)
            fp_mask = (mask == 0) & (full_prediction == 255)
            tn_mask = (mask == 0) & (full_prediction == 0)
            fn_mask = (mask == 255) & (full_prediction == 0)

            overlay = np.zeros_like(image)
            overlay[tp_mask] = colors["TP"]
            overlay[fp_mask] = colors["FP"]
            overlay[tn_mask] = colors["TN"]
            overlay[fn_mask] = colors["FN"]

            combined_overlay = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
            combined_image = np.hstack((
                image,
                cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(full_prediction, cv2.COLOR_GRAY2BGR),
                combined_overlay
            ))

            cv2.imwrite(os.path.join(output_dir, f"result_{idx}_{img_name}"), combined_image)

    pd.DataFrame(metrics).to_csv(os.path.join(output_dir, "iou_metrics.csv"), index=False)


if __name__ == "__main__":
    images_dir = my_utils.config.TEST_IMAGES_DIR
    masks_dir = my_utils.config.TEST_MASKS_DIR
    output_dir = "C:/Users/mirom/Desktop/preds_temp"

    transform = A.Compose([
        A.Resize(PATCH_SIZE, PATCH_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

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

    save_predictions_on_patches(model, images_dir, masks_dir, output_dir, transform, threshold=0.55)
