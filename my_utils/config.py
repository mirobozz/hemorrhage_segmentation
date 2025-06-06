import torch

ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['hemorrhage']
ACTIVATION = 'sigmoid'
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 210
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_IMAGES_DIR =''
TRAIN_MASKS_DIR =''
VAL_IMAGES_DIR =''
VAL_MASKS_DIR =''
TEST_IMAGES_DIR =''
TEST_MASKS_DIR =''

