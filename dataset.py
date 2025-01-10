import os
from torch.utils.data import Dataset
import numpy as np
from helper_functions.utils.image_utils import (
    load_rgb,
    load_grayscale
    )

class SegmentationDataset(Dataset):
    
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.jpg', '.png'))
        
        image = load_rgb(img_path)
        mask = load_grayscale(mask_path)
        
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask