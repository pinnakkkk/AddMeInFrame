import pytorch_lightning as pl
import argparse
import torch
import yaml
from dataset import SegmentationDataset
from pathlib import Path
from typing import Dict
from albumentations.core.serialization import from_dict
from torch.utils.data import DataLoader
from pytorch_toolbelt.losses import JaccardLoss, BinaryFocalLoss
from pytorch_lightning.loggers import WandbLogger

from helper_functions.config_parsing.utils import object_from_dict
from helper_functions.dl.pytorch.utils import state_dict_from_disk
from helper_functions.dl.pytorch.lightning import find_average

from utils import get_samples
from metrics import binary_mean_iou

train_path: Path = None
val_path: Path = None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=Path, required= True, help="path to the Hparams config")
    parser.add_argument("-t_path", "--training_path", type=Path, required=True)
    parser.add_argument("-v_path", "--valuation_path", type=Path, required=True)
    return parser.parse_args()


class SegmentPeople(pl.LightningModule):
    
    def __init__(self, hparams: Dict, device: str):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.curr_device = device
        self.validation_step_outputs = []

        self.model = object_from_dict(hparams["model"])
        if "resume_from_checkpoint" in hparams:
            correction: Dict[str, str] = {
                "model", ""
            }
            
            state_dict = state_dict_from_disk(
               file_path=self.hparams["resume_from_checkpoint"],
               device=self.device,
               rename_in_layers=correction
            )
        
            self.model.load_state_dict(state_dict)
            
        self.losses = [
            ("jaccard", 0.1, JaccardLoss(mode="binary", from_logits=True)),
            ("binnaryFocal", 0.9, BinaryFocalLoss())
        ]
            
    def forward(self, batch: torch.Tensor) -> torch.tensor:
        return self.model(batch)
        
    def setup(self, stage=None):
        self.train_samples = []
        
        for dataset_name in self.hparams["train_datasets"]:
            self.train_samples += get_samples(
                image_path = train_path/dataset_name/"images",
                mask_path = train_path/dataset_name/"labels"
            )
            
        self.val_samples = []
        self.val_dataset_names = {}
        
        for idx, dataset_name in enumerate(self.hparams["val_datasets"]):
            self.val_dataset_names[idx] = dataset_name
        
        for dataset_name in self.hparams["val_datasets"]:
            self.val_samples += [get_samples(
                image_path = val_path/dataset_name/"images",
                mask_path = val_path/dataset_name/"labels"
            )]

        self.to(self.curr_device)
     
    def train_dataloader(self):
        train_aug = from_dict(self.hparams["train_aug"])
            
        result = DataLoader(
         dataset=SegmentationDataset(self.train_samples, train_aug),
         batch_size=self.hparams["train_parameters"]["batch_size"],
         num_workers=self.hparams["num_workers"],
         shuffle=True,
         pin_memory=True,
         drop_last=True,
        )
        return result

    def val_dataloader(self):
        val_aug = from_dict(self.hparams["val_aug"])

        result = []

        for val_samples in self.val_samples:
            result += [
                DataLoader(
                    SegmentationDataset(val_samples, val_aug, length=None),
                    batch_size=self.hparams["val_parameters"]["batch_size"],
                    num_workers=self.hparams["num_workers"],
                    shuffle=False,
                    pin_memory=True,
                    drop_last=False,
                )
            ]
        return result
    
    def configure_optimizers(self):
        
        optimizer = object_from_dict(
            self.hparams["optimizer"],
            params= [param for param in self.model.parameters() if param.requires_grad]
        )
    
        scheduler = object_from_dict(self.hparams["scheduler"], optimizer=optimizer)
        self.optimizers = [optimizer]

        return self.optimizers, [scheduler]
    
    def training_step(self, batch, batch_idx):
        features = batch["features"]
        masks = batch["masks"]
        
        logits = self.forward(features)
        
        total_loss = 0
        logs = {}
        
        for loss_name, weight, loss in self.losses:
            
            cal_loss = loss(logits, masks)
            
            total_loss += cal_loss * weight
            
            logs[f"train_mask ({loss_name})"] = cal_loss
        
        logs["train_loss"] = total_loss
        
        logs["lr"] = self._get_current_lr()
        
        return {"loss": total_loss, "log": logs}
        
    def _get_current_lr(self) -> torch.Tensor:
        lrs = [optimizer.state_dict()["param_groups"][0]["lr"] for optimizer in self.optimizers]
        return torch.tensor(lrs[0], device=self.curr_device)
    
    def validation_step(self, batch, batch_id, dataloader_idx=0):
        features = batch["features"]
        masks = batch["masks"]

        logits = self.forward(features)

        result = {}
        total_loss = 0
        for loss_name, _, loss in self.losses:
            val_loss = loss(logits, masks)
            result[f"val_mask_{loss_name}"] = val_loss
            total_loss += val_loss
        
        # Log the overall validation loss
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        
        dataset_type = self.val_dataset_names[dataloader_idx]
        result[f"{dataset_type}_val_iou"] = binary_mean_iou(logits, masks)

        # Log other validation metrics if needed
        self.log(f'{dataset_type}_val_iou', result[f"{dataset_type}_val_iou"], on_epoch=True, prog_bar=True)

        self.validation_step_outputs.append(result)
        
        return result
    
    def on_validation_epoch_end(self):
        logs = {"epoch": self.trainer.current_epoch}
        
        for output_id, output in enumerate(self.validation_step_outputs):
            dataset_type = self.val_dataset_names[output_id]
            print(output.keys())
            print(output['vistas_val_iou'])
            avg_val_iou = find_average(output, f"{dataset_type}_val_iou")

            logs[f"{dataset_type}_val_iou"] = avg_val_iou
        
        return {"val_iou": avg_val_iou, "log": logs}
    
def main():
    args = get_args()
    
    global train_path, val_path 
    train_path = args.training_path
    val_path = args.valuation_path 
    
    if torch.cuda.is_available():
        device = "gpu"
    elif torch.mps.is_available():
        device = "mps"
    else: 
        device = "cpu"

    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    pipeline = SegmentPeople(hparams, device)

    Path(hparams["checkpoint_callback"]["dirpath"]).mkdir(exist_ok=True, parents=True)

    trainer = object_from_dict(
        hparams["trainer"],
        logger=WandbLogger(hparams["experiment_name"]),
        callbacks=object_from_dict(hparams["checkpoint_callback"]),
        **{"accelerator":device}
    )

    trainer.fit(pipeline)

if __name__ == "__main__":
    main()