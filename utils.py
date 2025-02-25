from pathlib import Path
from typing import Union, Dict, List, Tuple


def get_id2_file_paths(path: Union[str, Path]) -> Dict[str, Path]:
    return {file.stem: file for file in Path(path).glob("*.*")}


def get_samples(
    image_path: Path, 
    mask_path: Path) -> List[Tuple[Path, Path]]:

    image2path = get_id2_file_paths(image_path)
    mask2path = get_id2_file_paths(mask_path)
    
    resullt = [(image_file_path, mask2path[idx]) for idx, image_file_path in image2path.items()]
    return resullt