import os
import cv2
import numpy as np
import math

original_images_path = 'D:/bp_dataset/ieee_base/train/images'
original_masks_path = 'D:/bp_dataset/ieee_base/train/masks'

output_image_dir = 'D:/bp_dataset/ieee_clipper/train/images'
output_mask_dir = 'D:/bp_dataset/ieee_clipper/train/masks'

patch_size = 256

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

def resize_to_multiple_down(image, multiple):
    h, w = image.shape[:2]
    new_h = (h // multiple) * multiple
    new_w = (w // multiple) * multiple
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image

def save_patches(image, mask, patch_size, output_image_dir, output_mask_dir, base_name):
    h, w = image.shape[:2]
    patch_id = 0
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            img_patch = image[y:y + patch_size, x:x + patch_size]
            mask_patch = mask[y:y + patch_size, x:x + patch_size]

            img_patch_name = f"{base_name}_patch_{patch_id}.jpg"
            mask_patch_name = f"{base_name}_patch_{patch_id}.png"

            cv2.imwrite(os.path.join(output_image_dir, img_patch_name), img_patch, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            cv2.imwrite(os.path.join(output_mask_dir, mask_patch_name), mask_patch)

            patch_id += 1

def main():
    image_files = [f for f in os.listdir(original_images_path) if f.lower().endswith('.jpg')]

    for image_name in image_files:
        base_name = os.path.splitext(image_name)[0]
        image_path = os.path.join(original_images_path, image_name)
        mask_path = os.path.join(original_masks_path, base_name + '_HE.tif')

        if not os.path.exists(mask_path):
            print(f"Mask not found for {image_name}, skipping.")
            continue

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Error loading image or mask for {image_name}")
            continue

        resized_image = resize_to_multiple_down(image, patch_size)
        resized_mask = resize_to_multiple_down(mask, patch_size)

        save_patches(resized_image, resized_mask, patch_size, output_image_dir, output_mask_dir, base_name)

if __name__ == "__main__":
    main()
