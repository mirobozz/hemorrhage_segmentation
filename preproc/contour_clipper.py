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


def get_clip(image, mask, contour, clip_size):
    x, y, w, h = cv2.boundingRect(contour)

    center_x = int(x + w / 2)
    center_y = int(y + h / 2)

    start_x = center_x - int(clip_size / 2)
    start_y = center_y - int(clip_size / 2)

    if start_x < 0:
        start_x = 0
    if (start_x + clip_size) > image.shape[1]:
        start_x = image.shape[1] - clip_size

    if start_y < 0:
        start_y = 0
    if (start_y + clip_size) > image.shape[0]:
        start_y = image.shape[0] - clip_size

    clipped_image = image[start_y:start_y + clip_size, start_x:start_x + clip_size]
    clipped_mask = mask[start_y:start_y + clip_size, start_x:start_x + clip_size]

    return clipped_image, clipped_mask


def run_clipper():
    for mask_name in os.listdir(original_masks_path):
        if mask_name.endswith(".png"):

            mask_path = os.path.join(original_masks_path, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            image_name = os.path.splitext(mask_name)[0] + '.jpg'
            image_path = os.path.join(original_images_path, image_name)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Warning: Image {image_path} could not be loaded.")
                continue
            if mask is None:
                print(f"Warning: Mask {mask_path} could not be loaded.")
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            clips_created = 0
            for i, contour in enumerate(contours):
                if clips_created >= num_clips:
                    break

                clipped_image, clipped_mask = get_clip(image, mask, contour, clip_size)

                img_clip_name = f"{os.path.splitext(image_name)[0]}_clip_{clips_created}.png"
                mask_clip_name = f"{os.path.splitext(mask_name)[0]}_clip_{clips_created}.png"

                cv2.imwrite(os.path.join(clipped_images_path, img_clip_name), clipped_image)
                cv2.imwrite(os.path.join(clipped_masks_path, mask_clip_name), clipped_mask)
                print(f"Saved {img_clip_name} and {mask_clip_name}")

                clips_created += 1

if __name__ == "__main__":
    run_clipper()
