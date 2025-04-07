import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2


class SegmentationDataset(Dataset):

    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Image not loaded: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Mask not loaded: {mask_path}")

        mask = (mask > 0).astype('float32')

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, torch.tensor(mask).unsqueeze(0)


def create_dataloader(images_dir, masks_dir, batch_size, transform, shuffle=True):
    dataset = SegmentationDataset(images_dir, masks_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
