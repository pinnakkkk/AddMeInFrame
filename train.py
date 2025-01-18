import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
import os
from helper_functions.utils.image_utils import load_rgb
import segmentation_models_pytorch as smp
from utils import (
    save_checkpoint,
    load_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cpu"  # default
BATCH_SIZE = 25
NUM_EPOCHS = 4
NUM_WORKERS = 4
IMAGE_HEIGHT = 512  # 1280 originally
IMAGE_WIDTH = 512  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False

currPath = os.path.abspath(".")
TRAIN_IMG_DIR = currPath + "/Processed-DataSet/train_images/"
TRAIN_MASK_DIR = currPath + "/Processed-DataSet/train_labels/"
VAL_IMG_DIR = currPath + "/Processed-DataSet/val_images/"
VAL_MASK_DIR = currPath + "/Processed-DataSet/val_labels/"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    
    loop = tqdm(loader) # one epoch in the loader
    
    for batch_idx, (data, targets) in enumerate(loop):
        
        data = data.to(DEVICE)
        targets = targets.float().unsqueeze(1).to(DEVICE)
        
        # forward pass
        if scaler is None:
            # slow on cpu with the float operations
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        
        else:
            with torch.amp.autocast(device_type=DEVICE):
                predictions = model(data)
                loss = loss_fn(predictions, targets)
        
        # backward pass
        optimizer.zero_grad()   # flush the previous gradients 
        
        if scaler is None:
            loss.backward()
            optimizer.step()
        
        else:        
            scaler.scale(loss).backward()  # sclae the loss to avoid overflow and underflow and then calc the backpropogation using the chain rule of calculus
            scaler.step(optimizer)  # set the calc weights in the graph
            scaler.update() # update the internal scaler value for the next interation
        
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        

def main():
    global DEVICE
    
    # Set DEVICE
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(3, 1).to(DEVICE)
    
    # model = smp.Unet(
    #     encoder_name="timm-efficientnet-b3",        
    #     encoder_weights="noisy-student",     
    #     in_channels=3,                
    #     classes=1  
    # ).to(DEVICE)
    
    loss_fn = nn.BCEWithLogitsLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
     
    if DEVICE == 'cuda':
        scaler = torch.amp.GradScaler()    #creating a grad scaler 
    else:
        scaler = None
        
    if LOAD_MODEL:
        load_checkpoint(torch.load("unet_test1.pth.tar"), model, optimizer)
    
    for epoch in range(NUM_EPOCHS):
        
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        # save checkpoint
        checkpoint = {
            "state_dict" : model.state_dict(),
            "optimizer" : optimizer.state_dict()
        }
        
        save_checkpoint(checkpoint, "unet_test1.pth.tar")
        
        check_accuracy(val_loader, model, device=DEVICE)
        
        save_predictions_as_imgs(val_loader, model, DEVICE)
        
    model.eval()
    test_image = load_rgb("bg.png")
    # First normalize and convert to tensor
    test_transform = A.Compose([
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    # Apply transforms
    transformed = test_transform(image=test_image)
    test_image = transformed["image"]
    print(test_image.shape)
    test_image = test_image.unsqueeze(0).to(DEVICE)            
    print(test_image.shape)
    
    with torch.no_grad():
            preds = torch.sigmoid(model(test_image))
            preds = model(test_image)
            preds = (preds > 0.5).float()
            torchvision.utils.save_image(preds, f"{'saved_images'}/pred.png")
     
if __name__ == "__main__":
    main()