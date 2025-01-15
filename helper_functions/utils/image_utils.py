import cv2
import numpy as np
from typing import Union
from pathlib import Path


def load_rgb(image_path: Union[Path, str]) -> np.array:
    if Path(image_path).is_file():
        image_arr = cv2.imread(str(image_path))
        image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)
        
        return image_arr

    raise FileNotFoundError("File not found {image_path}")


def load_grayscale(image_path: Union[Path, str]) -> np.array:
    
    if Path(image_path).is_file():
        image_arr = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        return image_arr
    
    raise FileNotFoundError("File not found {image_path}")