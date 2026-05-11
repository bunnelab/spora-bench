import os
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from spora_io import MultiplexImagingDataset
from tqdm import tqdm


def compute_tile_tokens(config):

    spora_model = instantiate(config.model)

    for dataset_key in config.datasets:
        logger.info(f'Processing dataset: {dataset_key}')
        dataset_config = config.datasets[dataset_key]
        dataset_name = dataset_config.name

        dataset = MultiplexImagingDataset(
            name=dataset_name,
            path=dataset_config.path,
            modality=dataset_config.modality,
            standardization=dataset_config.standardization,
            resolution=dataset_config.resolution,
            tile_size=None,
            filter_list=['gaussian_blur',],
            use_mean_std=dataset_config.use_mean_std,
            disable_quantile_mask=dataset_config.disable_quantile_mask,
        )

        tile_dict = {}

        for tissue_id in tqdm(dataset.get_tissue_ids(kind="modality"), desc=f"Computing tile tokens for dataset '{dataset_name}'"):
            tile_tokens = spora_model.embed_tissue(dataset, tissue_id)
            if tile_tokens.shape[0] == 0:
                logger.warning(f"Tissue {tissue_id} has no valid tiles. Skipping.")
                continue
            tile_dict[tissue_id] = tile_tokens
        save_tile_tokens(tile_dict, config.output_dir, dataset_name, config.model.model_name)


def save_tile_tokens(tile_dict, output_dir, dataset_name, model_name):
    output_path = os.path.join(output_dir, model_name,  "tile_tokens", dataset_name)
    os.makedirs(output_path, exist_ok=True)
    for tissue_id, tile_tokens in tile_dict.items():
        np.save(os.path.join(output_path, f"{tissue_id}.npy"), tile_tokens)


def load_tile_tokens(input_dir, dataset_name, model_name):
    tile_dict = {}
    input_path = os.path.join(input_dir, model_name,  "tile_tokens", dataset_name)
    if not os.path.exists(input_path):
        logger.warning(f"No precomputed tile tokens found for dataset '{dataset_name}' at {input_path}.")
        return tile_dict

    for file in tqdm(os.listdir(input_path), desc=f"Loading tile tokens for dataset '{dataset_name}'"):
        if file.endswith(".npy"):
            tissue_id = Path(file).stem  # Remove .npy extension
            tile_tokens = np.load(os.path.join(input_path, file))
            tile_dict[tissue_id] = torch.from_numpy(tile_tokens)
    return tile_dict

if __name__ == "__main__":
    cli_config = OmegaConf.from_cli()

    base_config = OmegaConf.load('configs/base_config.yaml')
    model_config = OmegaConf.load(cli_config.model_config)
    datasets_config = OmegaConf.load(cli_config.datasets_config)

    config = OmegaConf.merge(base_config, model_config, datasets_config, cli_config)
    logger.info(f'Merged config:\n{OmegaConf.to_yaml(config)}')

    tile_dict = compute_tile_tokens(config)