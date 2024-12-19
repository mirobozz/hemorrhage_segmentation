import os
import my_utils.config as cfg

image_dir = cfg.TRAIN_IMAGES_DIR
mask_dir = cfg.TRAIN_MASKS_DIR

image_files = sorted(os.listdir(image_dir))
mask_files = sorted(os.listdir(mask_dir))

for img_file, mask_file in zip(image_files, mask_files):
    if os.path.splitext(img_file)[0] != os.path.splitext(mask_file)[0]:
        print(f"-: {img_file} | {mask_file}")
    else:
        print(f"+: {img_file} | {mask_file}")
