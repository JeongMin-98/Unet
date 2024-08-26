import logging
import random

import cv2
from PIL import Image
import numpy as np
import torch

from torch.utils.data import Dataset


class XrayDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None) -> None:
        super().__init__()
        # bone, background 0, 1
        self.num_classes = 2

        self.is_train = is_train
        self.root = cfg.DATASET.ROOT
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_type = cfg.DATASET.DATA_TYPE

        self.scale = cfg.DATASET.SCALE_FACTOR

        self.image_size = cfg.MODEL.IMAGE_SIZE

        # Unet type vs COCO type
        self.image_db = []
        self.mask_db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self, ):
        assert len(self.image_db) == len(self.mask_db), f'Image and mask {self.image_set} should be the same'
        return len(self.db)

    def __getitem__(self, index):
        # reference sbl code
        return super().__getitem__(index)
