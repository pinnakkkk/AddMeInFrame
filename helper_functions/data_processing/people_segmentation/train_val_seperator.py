from pathlib import Path
import shutil
import os

IMAGES_DIR = "/Users/lazylinuxer/MachineLearning/Google's-Addme/data/images/"
MASKS_DIR = "/Users/lazylinuxer/MachineLearning/Google's-Addme/data/labels/"

BASE_DIR = "/Users/lazylinuxer/MachineLearning/Google's-Addme/Processed-DataSet/"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1 

def main():
    if Path(IMAGES_DIR).exists and Path(MASKS_DIR).exists:
        
        images_list = os.listdir(Path(IMAGES_DIR))
        
        total_size = len(images_list)
        
        train_size = int(total_size * TRAIN_RATIO)
        val_size = int(total_size * VAL_RATIO)

        train_images = images_list[:train_size]
        train_masks = images_list[:train_size]
        
        val_images = images_list[train_size:train_size + val_size]
        val_masks = images_list[train_size:train_size + val_size]
        
        test_images = images_list[train_size + val_size:]
        test_masks = images_list[train_size + val_size:]
                
        train_images_output_folder =  BASE_DIR + "train_images"
        Path(train_images_output_folder).mkdir(exist_ok=True, parents=True)
    
        train_masks_output_folder = BASE_DIR + "train_labels"
        Path(train_masks_output_folder).mkdir(exist_ok=True, parents=True)
        
        val_images_output_folder =  BASE_DIR + "val_images"
        Path(val_images_output_folder).mkdir(exist_ok=True, parents=True)
    
        val_masks_output_folder = BASE_DIR + "val_labels"
        Path(val_masks_output_folder).mkdir(exist_ok=True, parents=True)

        test_images_output_folder =  BASE_DIR + "test_images"
        Path(test_images_output_folder).mkdir(exist_ok=True, parents=True)
    
        test_masks_output_folder = BASE_DIR + "test_labels"
        Path(test_masks_output_folder).mkdir(exist_ok=True, parents=True)
        
        for idx in range (0, len(train_images)):
            
            train_img = Path(Path.joinpath(Path(IMAGES_DIR), train_images[idx]))
            train_mask = Path(Path.joinpath(Path(MASKS_DIR), train_masks[idx].replace('.jpg', '.png')))
            
            shutil.copy(
                train_img, 
                str(train_images_output_folder))
            
            shutil.copy(
                train_mask, 
                str(train_masks_output_folder))
            
        
        for idx in range (0, len(val_images)):
            
            val_img = Path(Path.joinpath(Path(IMAGES_DIR), val_images[idx]))
            val_mask = Path(Path.joinpath(Path(MASKS_DIR), val_masks[idx].replace('.jpg', '.png')))
            
            shutil.copy(
                val_img, 
                str(val_images_output_folder))
            
            shutil.copy(
                val_mask, 
                str(val_masks_output_folder))
       
        
        for idx in range (0, len(val_images)):
            
            test_img = Path(Path.joinpath(Path(IMAGES_DIR), test_images[idx]))
            test_mask = Path(Path.joinpath(Path(MASKS_DIR), test_masks[idx].replace('.jpg', '.png')))
            
            shutil.copy(
                test_img, 
                str(test_images_output_folder))
            
            shutil.copy(
                test_mask, 
                str(test_masks_output_folder))
             
if __name__ == "__main__":
    main()