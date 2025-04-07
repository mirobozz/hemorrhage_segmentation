import os
import cv2
import numpy as np

original_images_path = 'D:/bp_dataset/combined_subsets/valid_test/images'
original_masks_path = 'D:/bp_dataset/combined_subsets/valid_test/masks'

clipped_base_path = 'D:/bp_dataset/sizeclip_aug/valid_test_sizeclip_aug/'
big_clipped_images_path = os.path.join(clipped_base_path, 'large_lesions/images')
big_clipped_masks_path = os.path.join(clipped_base_path, 'large_lesions/masks')
small_clipped_images_path = os.path.join(clipped_base_path, 'small_lesions/images')
small_clipped_masks_path = os.path.join(clipped_base_path, 'small_lesions/masks')

clip_size = 512
num_clips = 5
size_threshold = 5000


os.makedirs(big_clipped_images_path, exist_ok=True)
os.makedirs(big_clipped_masks_path, exist_ok=True)
os.makedirs(small_clipped_images_path, exist_ok=True)
os.makedirs(small_clipped_masks_path, exist_ok=True)


def get_clip(image, mask, contour, clip_size):
    x, y, w, h = cv2.boundingRect(contour)
    center_x = x + w // 2
    center_y = y + h // 2

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

        if not contours:
            print(f"Warning: No contours found in {mask_name}")
            continue

        clips_created = 0
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            if clips_created >= num_clips:
                break

            area = cv2.contourArea(contour)
            if area > size_threshold:
                img_dir, mask_dir = big_clipped_images_path, big_clipped_masks_path
            else:
                img_dir, mask_dir = small_clipped_images_path, small_clipped_masks_path

            clipped_image, clipped_mask = get_clip(image, mask, contour, clip_size)

            img_clip_name = f"{os.path.splitext(image_name)[0]}_clip_{clips_created}.png"
            mask_clip_name = f"{os.path.splitext(mask_name)[0]}_clip_{clips_created}.png"

            cv2.imwrite(os.path.join(img_dir, img_clip_name), clipped_image)
            cv2.imwrite(os.path.join(mask_dir, mask_clip_name), clipped_mask)
            print(f"Saved {img_clip_name} and {mask_clip_name} to {'big_lesions' if area > size_threshold else 'small_lesions'}")

            clips_created += 1


if __name__ == "__main__":
    run_clipper()
