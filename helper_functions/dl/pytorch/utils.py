import torch
from pathlib import Path
from typing import Any, Dict, Optional, Union


def rename_layers(
    correction: Dict[str, str],
    dict: Dict[str, Any]
) -> Dict[str, Any]:
    
    res_dict = dict.copy()
    for old_layer_name, new_layer_name in correction.items():
        for key in dict.keys():
            c_key = key.replace(old_layer_name, new_layer_name)
            res_dict[c_key] = res_dict.pop(key)
                
    return res_dict

def state_dict_from_disk(
file_path: Union[Path, str],
device: str,
rename_in_layers: Dict[str, str]
) -> Dict[str, Any]:
    
    checkpoints = torch.load(file_path, map_location=device)
    
    if "state_dict" in checkpoints:
        state_dict = checkpoints["state_dict"]
    else:
        state_dict = checkpoints
    
    if rename_in_layers is not None:
        rename_layers(correction=rename_in_layers, dict=state_dict)
    
    return state_dict
        