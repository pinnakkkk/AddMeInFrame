from pathlib import Path
import shutil
import os
import argparse

images_path = None
labels_path = None
output_path = None

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images_path", type=Path, required=True, help="Path to input data folder")
    parser.add_argument("-l", "--labels_path", type=Path, required=True, help="Path to labels data folder")
    parser.add_argument("-o", "--output_path", type=Path, required=True, help="Path to output data folder")
    return parser.parse_args()

def main():
    args = get_args()
    
    global images_path, labels_path, output_path
    images_path = args.images_path
    labels_path = args.labels_path
    output_path = args.output_path

    if images_path.exists and labels_path.exists:
        
        images_list = os.listdir(images_path)
        
        total_size = len(images_list)
        
        train_size = int(total_size * TRAIN_RATIO)
        val_size = int(total_size * VAL_RATIO)

        train_images = images_list[:train_size]
        train_masks = images_list[:train_size]
        
        val_images = images_list[train_size:train_size + val_size]
        val_masks = images_list[train_size:train_size + val_size]
        
        test_images = images_list[train_size + val_size:]
        test_masks = images_list[train_size + val_size:]
                
        train_images_output_path =  output_path/"train"/"images"
        train_images_output_path.mkdir(exist_ok=True, parents=True)
    
        train_masks_output_folder = output_path/"train"/"labels"
        train_masks_output_folder.mkdir(exist_ok=True, parents=True)
        
        val_images_output_folder =  output_path/"val"/"images"
        val_images_output_folder.mkdir(exist_ok=True, parents=True)
    
        val_masks_output_folder = output_path/"val"/"labels"
        val_masks_output_folder.mkdir(exist_ok=True, parents=True)

        test_images_output_folder =  output_path/"test"/"images"
        test_images_output_folder.mkdir(exist_ok=True, parents=True)
    
        test_masks_output_folder = output_path/"test"/"labels"
        test_masks_output_folder.mkdir(exist_ok=True, parents=True)
        
        for idx in range (0, len(train_images)):
            
            train_img = images_path/train_images[idx]
            train_mask = labels_path/train_masks[idx].replace('.jpg', '.png')
            
            shutil.copy(
                train_img, 
                str(train_images_output_path))
            
            shutil.copy(
                train_mask, 
                str(train_masks_output_folder))
            
        for idx in range (0, len(val_images)):
            
            val_img = images_path/val_images[idx]
            val_mask = labels_path/val_masks[idx].replace('.jpg', '.png')
            
            shutil.copy(
                val_img, 
                str(val_images_output_folder))
            
            shutil.copy(
                val_mask, 
                str(val_masks_output_folder))
       
        
        for idx in range (0, len(val_images)):
            
            test_img = images_path/test_images[idx]
            test_mask = labels_path/test_masks[idx].replace('.jpg', '.png')
            
            shutil.copy(
                test_img, 
                str(test_images_output_folder))
            
            shutil.copy(
                test_mask, 
                str(test_masks_output_folder))
             
if __name__ == "__main__":
    main()