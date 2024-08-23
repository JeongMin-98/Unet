# --------------------------------------------------------
# Reference from HRNet-Human-Pose-Estimation
# refact code from old one.
# Written by JeongMin Kim(jm.kim@dankook.ac.kr)
# ----------------------------------------------------

# from config import cfg
from .FCN import FCN
from .unet import UNet


def get_fcn(cfg, is_train, **kwargs):
    model = FCN(cfg)

    # if is_train:
    #     model.init_wegiht()

    return model


def get_UNet(cfg, is_train, **kwargs):
    model = UNet(1, 2)

    return model
