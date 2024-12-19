import os
import cv2
import numpy as np
import random

original_images_path = 'C:/Users/User/Desktop/bp/dataset/valid/images'
original_masks_path = 'C:/Users/User/Desktop/bp/dataset/valid/masks'
clipped_images_path = 'C:/Users/User/Desktop/bp/dataset/valid_augmented/images'
clipped_masks_path = 'C:/Users/User/Desktop/bp/dataset/valid_augmented/masks'

clip_size = 512
num_clips = 5

os.makedirs(clipped_images_path, exist_ok=True)
os.makedirs(clipped_masks_path, exist_ok=True)


def get_clip(image, mask, clip_size):
    height, width = image.shape[:2]

    x = random.randint(0, width - clip_size)
    y = random.randint(0, height - clip_size)

    clipped_image = image[y:y + clip_size, x:x + clip_size]
    clipped_mask = mask[y:y + clip_size, x:x + clip_size]

    return clipped_image, clipped_mask


def run_clipper():
    for image_name in os.listdir(original_images_path):
        if image_name.endswith((".jpg", ".png")):

            image_path = os.path.join(original_images_path, image_name)
            image = cv2.imread(image_path)

            mask_name = os.path.splitext(image_name)[0] + '.png'
            mask_path = os.path.join(original_masks_path, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Warning: Image {image_path} could not be loaded.")
                continue
            if mask is None:
                print(f"Warning: Mask {mask_path} could not be loaded.")
                continue

            for i in range(num_clips):
                clipped_image, clipped_mask = get_clip(image, mask, clip_size)

                img_clip_name = f"{os.path.splitext(image_name)[0]}_clip_{i}.png"
                mask_clip_name = f"{os.path.splitext(mask_name)[0]}_clip_{i}.png"

                cv2.imwrite(os.path.join(clipped_images_path, img_clip_name), clipped_image)
                cv2.imwrite(os.path.join(clipped_masks_path, mask_clip_name), clipped_mask)
                print(f"Saved {img_clip_name} and {mask_clip_name}")


if __name__ == "__main__":
    run_clipper()
