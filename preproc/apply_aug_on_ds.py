import os
import cv2
from tqdm import tqdm
import albumentations as A

# Папки
input_images_dir = 'D:/bp_dataset/sizeclip_blank_aug/valid_test_sizeclip_aug/small_lesions/images'         # путь к оригинальным изображениям
input_masks_dir = 'D:/bp_dataset/sizeclip_blank_aug/valid_test_sizeclip_aug/small_lesions/masks'           # путь к маскам (маски не изменяем)
output_images_dir = 'D:/bp_dataset/clahe_valid_test_sizeclip_aug/'  # путь, куда сохраняем CLAHE-версии

os.makedirs(output_images_dir, exist_ok=True)

# CLAHE трансформация
clahe_transform = A.Compose([
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True)
])

# Обработка всех изображений
image_filenames = sorted(os.listdir(input_images_dir))

for filename in tqdm(image_filenames, desc="Applying CLAHE"):
    img_path = os.path.join(input_images_dir, filename)
    image = cv2.imread(img_path)

    if image is None:
        print(f"⚠️ Не удалось загрузить: {img_path}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = clahe_transform(image=image)['image']
    augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

    out_path = os.path.join(output_images_dir, filename)
    cv2.imwrite(out_path, augmented)
