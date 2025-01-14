import argparse
import shutil
import cv2
import numpy as np

from pathlib import Path
from helper_functions.utils.image_utils import load_rgb
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading

PERSON_PIXELS = (220, 20, 60)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_folder", type=Path, help="Path to folder with images")
    parser.add_argument("-l", "--label_folder", type=Path, help="Path to folder with labels")
    parser.add_argument("-o", "--output_folder", type=Path, help="Path to the output folder")
    parser.add_argument("-s", "--sample_space", type=int, help="Limit the number of samples to process")
    parser.add_argument("-thd", "--num_of_thread", default=4, type=int, help="Number of threads")
    return parser.parse_args()

# global variables
num_of_labels_generated = 0
counter_lock = threading.Lock()
sample_limit_reached = threading.Event()

def process_single_image(args: argparse, label_file_name: Path, output_image_folder: Path, output_label_folder: Path) -> bool:
    global num_of_labels_generated
    
    # Check if sample limit is reached before processing
    if sample_limit_reached.is_set():
        return False
        
    label = load_rgb(label_file_name)
    mask = (label == PERSON_PIXELS).all(axis=-1).astype(np.uint8)
    
    if mask.sum() == 0:
        return False
    
    with counter_lock:
        if num_of_labels_generated >= args.sample_space:
            sample_limit_reached.set()
            return False
            
        shutil.copy(
            str(args.image_folder / f"{label_file_name.stem}.jpg"),
            str(output_image_folder / f"{label_file_name.stem}.jpg")
        )
        
        cv2.imwrite(str(output_label_folder/label_file_name.name), mask * 255)
        num_of_labels_generated += 1
        
    return True

def main():
    args = get_args()
    label_files = list(args.label_folder.rglob("*.png"))
    
    output_image_folder = args.output_folder / "images"
    output_image_folder.mkdir(exist_ok=True, parents=True)
    
    output_label_folder = args.output_folder / "labels"
    output_label_folder.mkdir(exist_ok=True, parents=True)
    
    with ThreadPoolExecutor(args.num_of_thread) as executor:
        futures = []
        for label_file_name in tqdm(sorted(label_files)):
            if sample_limit_reached.is_set():
                break
                
            future = executor.submit(
                process_single_image,
                args,
                label_file_name,
                output_image_folder,
                output_label_folder
            )
            futures.append(future)

        # Wait for ongoing tasks to complete
        for future in tqdm(futures):
            future.result()

    print(f"Final num_of_labels_generated: {num_of_labels_generated}")       
                
if __name__ == "__main__":
    main()