import random
import subprocess
from pathlib import Path

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

def check_hf_login():
    """
    Verifies that the current environment is authenticated with the Hugging Face Hub.
    Raises:
        RuntimeError: If no valid Hugging Face credentials are found.
    """
    from huggingface_hub import HfApi

    try:
        HfApi().whoami()
    except Exception as e:
        raise RuntimeError(
            "Hugging Face authentication is required to download this model's weights. "
            "Run `huggingface-cli login` (or set the HF_TOKEN environment variable) and retry."
        ) from e

def clone_git_repo(url: str, dest: Path):
    """
    Clones a git repository to `dest` if it does not already exist, raising an informative
    error if git is unavailable or the clone fails (e.g. missing network access or permissions).
    Args:
        url (str): The git repository URL to clone.
        dest (Path): The destination directory for the clone.
    """
    if dest.exists():
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git", "clone", url, str(dest)], check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("`git` executable not found. Please install git and ensure it is on PATH.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to clone '{url}' into '{dest}'. Check your network connection and git "
            f"credentials/permissions for this repository.\ngit stderr: {e.stderr}"
        ) from e
