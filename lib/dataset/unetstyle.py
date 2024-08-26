# -----------------------------------------------------------
# 
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------
import logging
import numpy as np
import os
import os.path as osp

from dataset.XrayDataset import XrayDataset
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path

from utils.tools import load_image


def unique_mask_values(idx, mask_dir):
    mask_dir = Path(mask_dir)
    mask_file = list(mask_dir.glob(f"{idx}_mask.*"))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class UNetStyleDataset(XrayDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        assert cfg.DATASET.DATA_TYPE == 'Unet', "Input Data is not UnetStyle."

        self.image_dir = osp.join(self.root, 'imgs')
        self.mask_dir = osp.join(self.root, 'masks')

        assert 0 < self.scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [osp.splitext(file)[0] for file in os.listdir(self.image_dir) if
                    osp.isfile(osp.join(self.image_dir, file)) and not file.startswith('.')]

        if not self.ids:
            raise RuntimeError(f'No input file found in {self.image_dir}, make sure you put your images there')

        logging.info(f"Creating dataset with {len(self.ids)} examples")
        logging.info("Scanning mask files to determine unique values")

        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir), self.ids),
                total=len(self.ids)
            ))

        self.maks_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f"Unique mask value: {self.maks_values}")
        
        self.image_db, self.mask_db = self._get_db()
        
        logging.info(f"Load {len(self.image_db)} samples")

    def _get_db(self):
        if self.is_train:
            # use ground truth mask.
            gt_img_db, gt_mask_db = self._load_img_and_mask()
        else:
            # inference
            pass
    
        return gt_img_db, gt_mask_db
    
    def _load_img_and_mask(self):
        gt_img_db, gt_mask_db = [], []
        for idx in self.ids:
            # file format use(wip)
            mask_file_path = osp.join(self.mask_dir, f"{idx}_mask.png")
            img_file_path = osp.join(self.image_dir, f"{idx}.jpg")
            
            gt_img_db.append(img_file_path)
            gt_mask_db.append(mask_file_path)
        
        return gt_img_db, gt_mask_db
    
                
            
            
            