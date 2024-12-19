import torch

ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['hemorrhage']
ACTIVATION = 'sigmoid'
LEARNING_RATE = 0.0001
BATCH_SIZE = 8
NUM_EPOCHS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_IMAGES_DIR = 'C:/Users/User/Desktop/bp/dataset/train_augmented/images'
TRAIN_MASKS_DIR = 'C:/Users/User/Desktop/bp/dataset/train_augmented/masks'
VAL_IMAGES_DIR = 'C:/Users/User/Desktop/bp/dataset/valid/images'
VAL_MASKS_DIR = 'C:/Users/User/Desktop/bp/dataset/valid/masks'
TEST_IMAGES_DIR = 'C:/Users/User/Desktop/bp/dataset/test/images'
TEST_MASKS_DIR = 'C:/Users/User/Desktop/bp/dataset/test/masks'
