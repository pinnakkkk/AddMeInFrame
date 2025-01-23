from typing import List

import torch

def find_average(outputs: List, name: str) -> torch.Tensor:
    
    if len(outputs[name].shape) == 0:
        print("output",type(outputs))
        return torch.stack([x[name] for x in outputs]).mean()
    return torch.cat([x[name] for x in outputs]).mean()