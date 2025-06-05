import os
import PIL.Image as Image
import numpy as np
import cv2

dataset_path = 'D:/bp_dataset'
images_path = os.path.join(dataset_path, 'imgs_tmp')
masks_path = os.path.join(dataset_path, 'msks_tmp')

results_path = 'D:/bp_dataset/ieee_base'
cropped_images_path = os.path.join(results_path, 'images')
cropped_masks_path = os.path.join(results_path, 'masks')

print("Файлы в папке масок:")
print(os.listdir(masks_path)[:10])



def crop_bb_by_image(image, mask, threshold=30):
    image_np = np.array(image)
    mask_np = np.array(mask)

    grayscale = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    non_zero_mask = grayscale > threshold  # теперь не просто > 0

    coords = np.argwhere(non_zero_mask)

    if coords.size == 0:
        return image_np, mask_np

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped_image = image_np[y_min:y_max + 1, x_min:x_max + 1]
    cropped_mask = mask_np[y_min:y_max + 1, x_min:x_max + 1]

    return cropped_image, cropped_mask



os.makedirs(cropped_images_path, exist_ok=True)
os.makedirs(cropped_masks_path, exist_ok=True)

for img_name in os.listdir(images_path):
    if img_name.lower().endswith(('.jpg', '.png', '.tif', '.tiff')):
        image_path = os.path.join(images_path, img_name)
        base_name = os.path.splitext(img_name)[0]

        matched_masks = [m for m in os.listdir(masks_path) if base_name in m]

        if len(matched_masks) == 0:
            print(f"[!] Маска не найдена для {img_name}")
            continue
        elif len(matched_masks) > 1:
            print(f"[!] Несколько масок найдено для {img_name}: {matched_masks}. Взята первая.")

        mask_name = matched_masks[0]
        mask_path = os.path.join(masks_path, mask_name)

        try:
            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')

            cropped_image, cropped_mask = crop_bb_by_image(image, mask)

            cropped_pil_image = Image.fromarray(cropped_image)
            cropped_pil_mask = Image.fromarray(cropped_mask)

            cropped_image_save_path = os.path.join(cropped_images_path, img_name)
            cropped_mask_save_path = os.path.join(cropped_masks_path, mask_name)

            cropped_pil_image.save(cropped_image_save_path)
            cropped_pil_mask.save(cropped_mask_save_path)

        except Exception as e:
            print(f"[!!] Ошибка с {img_name}: {e}")

