import torch

ENCODER = 'resnet18'
#ENCODER = 'resnet34'
#ENCODER = 'resnet152'
#ENCODER = 'efficientnet-b1'
#ENCODER = 'efficientnet-b7'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['hemorrhage']
ACTIVATION = 'sigmoid'
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
#BATCH_SIZE = 16
NUM_EPOCHS = 210
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

'''

#1 (ResNet34 shift)
TRAIN_IMAGES_DIR = 'D:/bp_dataset/lwnet/train_clean_images'
TRAIN_MASKS_DIR = 'D:/bp_dataset/lwnet/train_clean_masks'
VAL_IMAGES_DIR = 'D:/bp_dataset/sizeclip_aug/valid_sizeclip_aug/small_lesions/masks'
VAL_MASKS_DIR = 'D:/bp_dataset/sizeclip_aug/valid_sizeclip_aug/small_lesions/masks'
TEST_IMAGES_DIR = 'D:/bp_dataset/base_subsets/train/images'
TEST_MASKS_DIR = 'D:/bp_dataset/base_subsets/train/masks'


#2 (r34 vs r152 vs efnetb1)
TRAIN_IMAGES_DIR = 'D:/bp_dataset/sizeclip_blank_aug/valid_test_sizeclip_aug/small_lesions/images'
TRAIN_MASKS_DIR = 'D:/bp_dataset/sizeclip_blank_aug/valid_test_sizeclip_aug/small_lesions/masks'
VAL_IMAGES_DIR = 'D:/bp_dataset/lwnet/train_clean_images'
VAL_MASKS_DIR = 'D:/bp_dataset/lwnet/train_clean_masks'
TEST_IMAGES_DIR = 'D:/bp_dataset/base_subsets/train/images'
TEST_MASKS_DIR = 'D:/bp_dataset/base_subsets/train/masks'


#3 (clahe vs no clahe)
TRAIN_IMAGES_DIR = 'D:/bp_dataset/clahe_valid_test_sizeclip_aug'
TRAIN_MASKS_DIR = 'D:/bp_dataset/sizeclip_blank_aug/valid_test_sizeclip_aug/small_lesions/masks'
VAL_IMAGES_DIR = 'D:/bp_dataset/lwnet/train_clean_images'
VAL_MASKS_DIR = 'D:/bp_dataset/lwnet/train_clean_masks'
TEST_IMAGES_DIR = 'D:/bp_dataset/base_subsets/train/images'
TEST_MASKS_DIR = 'D:/bp_dataset/base_subsets/train/masks'



#4 loss types (resnet 152)

TRAIN_IMAGES_DIR = 'D:/bp_dataset/sizeclip_blank_aug/valid_test_sizeclip_aug/small_lesions/images'
TRAIN_MASKS_DIR = 'D:/bp_dataset/sizeclip_blank_aug/valid_test_sizeclip_aug/small_lesions/masks'
VAL_IMAGES_DIR = 'D:/bp_dataset/lwnet/train_clean_images'
VAL_MASKS_DIR = 'D:/bp_dataset/lwnet/train_clean_masks'
TEST_IMAGES_DIR = 'D:/bp_dataset/base_subsets/train/images'
TEST_MASKS_DIR = 'D:/bp_dataset/base_subsets/train/masks'


#5 test ieee dataseta
TRAIN_IMAGES_DIR = 'D:/bp_dataset/ieee_smartclipper/train/images'
TRAIN_MASKS_DIR = 'D:/bp_dataset/ieee_smartclipper/train/masks'
VAL_IMAGES_DIR = 'D:/bp_dataset/lwnet/train_clean_images'
VAL_MASKS_DIR = 'D:/bp_dataset/lwnet/train_clean_masks'
TEST_IMAGES_DIR = 'D:/bp_dataset/ieee_base/test/images'
TEST_MASKS_DIR = 'D:/bp_dataset/ieee_base/test/masks'



#6 unet++ resnet34 resnet152

TRAIN_IMAGES_DIR = 'D:/bp_dataset/sizeclip_blank_aug/valid_test_sizeclip_aug/small_lesions/images'
TRAIN_MASKS_DIR = 'D:/bp_dataset/sizeclip_blank_aug/valid_test_sizeclip_aug/small_lesions/masks'
VAL_IMAGES_DIR = 'D:/bp_dataset/lwnet/train_clean_images'
VAL_MASKS_DIR = 'D:/bp_dataset/lwnet/train_clean_masks'
TEST_IMAGES_DIR = 'D:/bp_dataset/base_subsets/train/images'
TEST_MASKS_DIR = 'D:/bp_dataset/base_subsets/train/masks'


#7 efnet7 unet small

TRAIN_IMAGES_DIR = 'D:/bp_dataset/sizeclip_blank_aug/valid_test_sizeclip_aug/small_lesions/images'
TRAIN_MASKS_DIR = 'D:/bp_dataset/sizeclip_blank_aug/valid_test_sizeclip_aug/small_lesions/masks'
VAL_IMAGES_DIR = 'D:/bp_dataset/lwnet/train_clean_images'
VAL_MASKS_DIR = 'D:/bp_dataset/lwnet/train_clean_masks'
TEST_IMAGES_DIR = 'D:/bp_dataset/base_subsets/train/images'
TEST_MASKS_DIR = 'D:/bp_dataset/base_subsets/train/masks'


#8 efnet7 unet++ small

TRAIN_IMAGES_DIR = 'D:/bp_dataset/sizeclip_blank_aug/valid_test_sizeclip_aug/small_lesions/images'
TRAIN_MASKS_DIR = 'D:/bp_dataset/sizeclip_blank_aug/valid_test_sizeclip_aug/small_lesions/masks'
VAL_IMAGES_DIR = 'D:/bp_dataset/lwnet/train_clean_images'
VAL_MASKS_DIR = 'D:/bp_dataset/lwnet/train_clean_masks'
TEST_IMAGES_DIR = 'D:/bp_dataset/base_subsets/train/images'
TEST_MASKS_DIR = 'D:/bp_dataset/base_subsets/train/masks'
'''

#9 resnet unet large bce 256 patch bs8

TRAIN_IMAGES_DIR = 'D:/bp_dataset/sizeclip_blank_aug/valid_test_sizeclip_aug/large_lesions/images'
TRAIN_MASKS_DIR = 'D:/bp_dataset/sizeclip_blank_aug/valid_test_sizeclip_aug/large_lesions/masks'
VAL_IMAGES_DIR = 'D:/bp_dataset/lwnet/train_clean_images'
VAL_MASKS_DIR = 'D:/bp_dataset/lwnet/train_clean_masks'
TEST_IMAGES_DIR = 'D:/bp_dataset/base_subsets/train/images'
TEST_MASKS_DIR = 'D:/bp_dataset/base_subsets/train/masks'