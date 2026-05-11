import random
import torch
import numpy as np
from glob import glob
from omegaconf import OmegaConf
from omegaconf import DictConfig
from typing import Union, List

def set_seed(seed: int):
    """
    Sets the random seed for reproducibility across various libraries.
    Args:
        seed (int): The seed value to set for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_multiple_configs(config_paths: Union[str, List[str]]) -> DictConfig:
    """
    Loads and merges a flexible number of configuration files.
    Args:
        config_paths (Union[str, List[str]]): A single config file path or a list of config file paths. Wildcards are supported in the paths.
    Returns:
        DictConfig: A merged OmegaConf object containing all configurations.
    """
    if isinstance(config_paths, str):
        config_paths = [config_paths]

    configs = []
    for path in config_paths:
        extended_paths = glob(path, recursive=True)  # Expand wildcards
        extended_paths = sorted(extended_paths)  # Sort paths for consistent loading order
        for extended_path in extended_paths:
            config = OmegaConf.load(extended_path)
            configs.append(config)

    if not configs:
        raise ValueError(f"No configuration files found for the provided paths: {config_paths}")            

    return OmegaConf.merge(*configs)