import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder_names, get_preprocessing_params
from torch import nn
from my_utils.config import *

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

model.to(DEVICE)
print(model)
print(get_encoder_names())
print(get_preprocessing_params("resnet34"))
