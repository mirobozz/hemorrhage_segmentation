import os
import cv2
import numpy as np
import random

original_images_path = 'D:/bp_dataset/ieee_base/train/images'
original_masks_path = 'D:/bp_dataset/ieee_base/train/masks'

clipped_base_path = 'D:/bp_dataset/ieee_sizeclip_blank_aug/train/'
big_clipped_images_path = os.path.join(clipped_base_path, 'large_lesions/images')
big_clipped_masks_path = os.path.join(clipped_base_path, 'large_lesions/masks')
small_clipped_images_path = os.path.join(clipped_base_path, 'small_lesions/images')
small_clipped_masks_path = os.path.join(clipped_base_path, 'small_lesions/masks')
random_clipped_images_path = os.path.join(clipped_base_path, 'random_clips/images')
random_clipped_masks_path = os.path.join(clipped_base_path, 'random_clips/masks')

clip_size = 256
num_clips = 10
num_random_clips_per_image = 2
size_threshold = 10000

os.makedirs(big_clipped_images_path, exist_ok=True)
os.makedirs(big_clipped_masks_path, exist_ok=True)
os.makedirs(small_clipped_images_path, exist_ok=True)
os.makedirs(small_clipped_masks_path, exist_ok=True)
os.makedirs(random_clipped_images_path, exist_ok=True)
os.makedirs(random_clipped_masks_path, exist_ok=True)

def get_clip(image, mask, center_x, center_y, clip_size):
    start_x = max(0, min(center_x - clip_size // 2, image.shape[1] - clip_size))
    start_y = max(0, min(center_y - clip_size // 2, image.shape[0] - clip_size))
    clipped_image = image[start_y:start_y + clip_size, start_x:start_x + clip_size]
    clipped_mask = mask[start_y:start_y + clip_size, start_x:start_x + clip_size]
    return clipped_image, clipped_mask

def run_clipper():
    for mask_name in os.listdir(original_masks_path):
        if not mask_name.endswith(".png"):
            continue

        mask_path = os.path.join(original_masks_path, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image_name = os.path.splitext(mask_name)[0] + '.jpg'
        image_path = os.path.join(original_images_path, image_name)
        image = cv2.imread(image_path)

        if image is None or mask is None:
            print(f"Warning: Could not load image {image_path} or mask {mask_path}")
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        clips_created = 0
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            if clips_created >= num_clips:
                break

            area = cv2.contourArea(contour)
            center_x, center_y, _, _ = cv2.boundingRect(contour)
            center_x += _ // 2
            center_y += _ // 2

            if area > size_threshold:
                img_dir, mask_dir = big_clipped_images_path, big_clipped_masks_path
            else:
                img_dir, mask_dir = small_clipped_images_path, small_clipped_masks_path

            clipped_image, clipped_mask = get_clip(image, mask, center_x, center_y, clip_size)

            img_clip_name = f"{os.path.splitext(image_name)[0]}_clip_{clips_created}.png"
            mask_clip_name = f"{os.path.splitext(mask_name)[0]}_clip_{clips_created}.png"

            cv2.imwrite(os.path.join(img_dir, img_clip_name), clipped_image)
            cv2.imwrite(os.path.join(mask_dir, mask_clip_name), clipped_mask)
            print(f"Saved {img_clip_name} and {mask_clip_name} to {'big_lesions' if area > size_threshold else 'small_lesions'}")

            clips_created += 1

        for i in range(num_random_clips_per_image):
            attempts = 0
            max_attempts = 10
            margin = 30

            while attempts < max_attempts:
                center_x = random.randint(margin + clip_size // 2, image.shape[1] - margin - clip_size // 2)
                center_y = random.randint(margin + clip_size // 2, image.shape[0] - margin - clip_size // 2)
                clipped_image, clipped_mask = get_clip(image, mask, center_x, center_y, clip_size)

                if np.sum(clipped_mask) == 0:
                    img_clip_name = f"{os.path.splitext(image_name)[0]}_random_{i}.png"
                    mask_clip_name = f"{os.path.splitext(mask_name)[0]}_random_{i}.png"

                    for img_dir, mask_dir in [
                        (big_clipped_images_path, big_clipped_masks_path),
                        (small_clipped_images_path, small_clipped_masks_path)
                    ]:
                        cv2.imwrite(os.path.join(img_dir, img_clip_name), clipped_image)
                        cv2.imwrite(os.path.join(mask_dir, mask_clip_name), clipped_mask)

                    print(f"Saved random empty clip {img_clip_name} to both big_lesions and small_lesions")
                    break

                attempts += 1


if __name__ == "__main__":
    run_clipper()
