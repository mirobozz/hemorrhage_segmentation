import os
import cv2
import numpy as np
import torch
import pandas as pd
from albumentations.pytorch import ToTensorV2
import albumentations as A
import segmentation_models_pytorch as smp
from scipy.ndimage import label
from my_utils.metrics import compute_f1

import my_utils.config

colors = {
    "TP": [0, 255, 0],
    "FP": [0, 0, 255],
    "TN": [255, 0, 0],
    "FN": [255, 193, 203],
}

PATCH_SIZE = 256

def compute_iou(pred, mask):
    pred = pred.astype(bool)
    mask = mask.astype(bool)
    intersection = np.logical_and(pred, mask).sum()
    union = np.logical_or(pred, mask).sum()
    return float('1.0') if union == 0 else intersection / union

def crop_bb_by_image(image, mask):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.argwhere(grayscale > 0)
    if coords.size == 0:
        return image, mask
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return image[y_min:y_max+1, x_min:x_max+1], mask[y_min:y_max+1, x_min:x_max+1]

def resize_to_nearest(image: np.ndarray, base: int = PATCH_SIZE) -> np.ndarray:
    h, w = image.shape[:2]
    new_h = (h // base) * base
    new_w = (w // base) * base
    if new_h == 0 or new_w == 0:
        raise ValueError(f"Invalid resize dimensions: {(new_w, new_h)} from input shape {(h, w)}")
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def split_into_patches(image: np.ndarray, patch_size: int = PATCH_SIZE):
    patches = []
    h, w = image.shape[:2]
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append((patch, (y, x)))
    return patches

def merge_patches(patches: dict, image_shape: tuple, patch_size: int = PATCH_SIZE):
    full_mask = np.zeros(image_shape, dtype=np.uint8)
    for (y, x), patch in patches.items():
        full_mask[y:y+patch_size, x:x+patch_size] = patch
    return full_mask

def filter_by_area(prediction, area_threshold):
    bin_pred = (prediction > 0.55).astype(np.uint8)
    labeled_array, num_features = label(bin_pred)
    large = np.zeros_like(bin_pred)
    small = np.zeros_like(bin_pred)

    for label_idx in range(1, num_features + 1):
        region = (labeled_array == label_idx)
        area = region.sum()
        if area >= area_threshold:
            large[region] = 1
        else:
            small[region] = 1

    return large, small

def save_predictions_dual_patch_or(model_large, model_small, images_dir, masks_dir, output_dir, transform_patch, alpha=0.5):
    os.makedirs(output_dir, exist_ok=True)
    model_large.eval()
    model_small.eval()
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
            image = resize_to_nearest(image)
            mask = resize_to_nearest(mask)

            patches = split_into_patches(image)
            final_patches = {}

            for patch, (y, x) in patches:
                aug_patch = transform_patch(image=patch)
                input_tensor = aug_patch["image"].unsqueeze(0).to(my_utils.config.DEVICE)

                out_large = torch.sigmoid(model_large(input_tensor))[0, 0].cpu().numpy()
                out_small = torch.sigmoid(model_small(input_tensor))[0, 0].cpu().numpy()

                bin_large = (out_large > 0.55).astype(np.uint8)
                bin_small = (out_small > 0.55).astype(np.uint8)

                combined_patch = np.maximum(bin_large, bin_small) * 255
                final_patches[(y, x)] = combined_patch.astype(np.uint8)

            final_prediction = merge_patches(final_patches, image.shape[:2])
            iou = compute_iou(final_prediction > 0, mask > 0)

            pred_tensor = torch.tensor((final_prediction > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)
            mask_tensor = torch.tensor((mask > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)
            f1 = compute_f1(pred_tensor, mask_tensor, threshold=0.5)

            metrics.append({"image_name": img_name, "IoU": iou, "F1": f1})

            tp = (mask == 255) & (final_prediction == 255)
            fp = (mask == 0) & (final_prediction == 255)
            tn = (mask == 0) & (final_prediction == 0)
            fn = (mask == 255) & (final_prediction == 0)

            overlay = np.zeros_like(image)
            overlay[tp] = colors["TP"]
            overlay[fp] = colors["FP"]
            overlay[tn] = colors["TN"]
            overlay[fn] = colors["FN"]

            combined_overlay = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
            combined_image = np.hstack((
                image,
                cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(final_prediction, cv2.COLOR_GRAY2BGR),
                combined_overlay
            ))

            cv2.imwrite(os.path.join(output_dir, f"dualpatch_or_{idx}_{img_name}"), combined_image)

    pd.DataFrame(metrics).to_csv(os.path.join(output_dir, "dualpatch_or_iou_metrics.csv"), index=False)


if __name__ == "__main__":
    images_dir = my_utils.config.TEST_IMAGES_DIR
    masks_dir = my_utils.config.TEST_MASKS_DIR
    output_dir = "C:/Users/mirom/Desktop/exp_unionbased"

    transform_patch = A.Compose([
        A.Resize(PATCH_SIZE, PATCH_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    transform_full = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    model_large = smp.Unet(
        encoder_name='resnet18',
        encoder_weights=None,
        classes=len(my_utils.config.CLASSES),
        activation=my_utils.config.ACTIVATION,
    )
    model_small = smp.Unet(
        encoder_name='resnet34',
        encoder_weights=None,
        classes=len(my_utils.config.CLASSES),
        activation=my_utils.config.ACTIVATION,
    )

    checkpoint_large = torch.load("../train/best_model.pth", map_location=my_utils.config.DEVICE)
    checkpoint_small = torch.load("../train/resnet34.pth", map_location=my_utils.config.DEVICE)

    model_large.load_state_dict(checkpoint_large["model_state_dict"])
    model_small.load_state_dict(checkpoint_small["model_state_dict"])

    model_large = model_large.to(my_utils.config.DEVICE)
    model_small = model_small.to(my_utils.config.DEVICE)

    save_predictions_dual_patch_or(model_large, model_small, images_dir, masks_dir, output_dir, transform_patch,
                                   alpha=0.5)
