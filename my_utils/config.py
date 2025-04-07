import torch

ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['hemorrhage']
ACTIVATION = 'sigmoid'
LEARNING_RATE = 0.00001
BATCH_SIZE = 8
NUM_EPOCHS = 500
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_IMAGES_DIR = 'D:/bp_dataset/sizeclip_aug/train_sizeclip_aug/large_lesions/images'
TRAIN_MASKS_DIR = 'D:/bp_dataset/sizeclip_aug/train_sizeclip_aug/large_lesions/masks'
VAL_IMAGES_DIR = 'D:/bp_dataset/base_subsets/train/images'
VAL_MASKS_DIR = 'D:/bp_dataset/base_subsets/train/masks'
TEST_IMAGES_DIR = 'D:/bp_dataset/sizeclip_aug/test_sizeclip_aug/small_lesions/images'
TEST_MASKS_DIR = 'D:/bp_dataset/sizeclip_aug/test_sizeclip_aug/small_lesions/masks'
