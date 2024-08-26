import logging
import random

import cv2
from PIL import Image
import numpy as np
import torch

from torch.utils.data import Dataset
from utils.tools import load_image


class XrayDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None) -> None:
        super().__init__()
        # bone, background 0, 1
        self.num_classes = 2
        self.mask_value = []

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

    def preprocess(self, task, is_mask):
        ipt_w, ipt_h = self.image_size
        # ipt_w, ipt_h = int(self.scale * ipt_w), int(self.scale * ipt_h)
        assert ipt_w > 0 and ipt_h > 0 , "Image is too small"
        task = task.resize((ipt_w, ipt_h), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        task = np.asarray(task)
        
        if is_mask:
            mask = np.zeros((ipt_w, ipt_h), dtype=np.int64)
            for i, v in enumerate(self.mask_values):
                if task.ndim == 2:
                    mask[task == v] = i
                else:
                    mask[(task == v).all(-1)] = i

            return mask

        else:
            if task.ndim == 2:
                img = task[np.newaxis, ...]
            else:
                img = task.transpose((2, 0, 1))

            if (task > 1).any():
                img = task / 255.0

            return img
        
    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self, ):
        assert len(self.image_db) == len(self.mask_db), f'Image and mask {self.image_set} should be the same'
        return len(self.db)

    def __getitem__(self, index):
        image_file_path = self.image_db[index]
        mask_file_path = self.mask_db[index]
        
        mask = load_image(image_file_path)
        img = load_image(image_file_path)

        
