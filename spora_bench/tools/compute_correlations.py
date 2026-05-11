import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import OmegaConf
from collections import defaultdict

from spora_bench.utils.setup_utils import load_multiple_configs, set_seed
from spora_io.datasets import MultiplexImagingDataset, MultiplexTissue
from tqdm import tqdm

def compute_correlations(config): 

    output_dir = Path(config.output_dir)
    results_dir = output_dir / 'dataset_statistics' 
    os.makedirs(results_dir, exist_ok=True)

    for dataset_key, dataset_config in config.datasets.items():
        logger.info(f'Processing dataset: {dataset_key}')
        dataset_name = dataset_config.name

        dataset = MultiplexImagingDataset(
            name=dataset_name,
            path=dataset_config.path,
            modality=dataset_config.modality,
            standardization=dataset_config.standardization,
            resolution=dataset_config.resolution,
            tile_size=128,
            tile_strategy='default',
            filter_list=['gaussian_blur',],
            use_mean_std=dataset_config.use_mean_std,
            disable_quantile_mask=dataset_config.disable_quantile_mask,
            verbose=False,
        )

        pairwise_correlations = defaultdict(list)

        channel_to_uniprot_map = dict(zip(dataset.channel_list['channel_name'], dataset.channel_list['uniprot_id']))
        test_tissue_ids = dataset.tissue_modality_metadata[dataset.tissue_modality_metadata['split'] == 'test'].index.values

        for tissue_id in tqdm(test_tissue_ids, desc=f"Processing tissues for dataset {dataset_name}"):
            tissue = dataset.get_tissue(tissue_id, kind="uniprot_filtered", preprocess=True, image_mode="CHW")
            
            channel_names = tissue.channel_names
            image = tissue.image # (C, H, W)
            image_flattened = image.reshape(image.shape[0], -1) # (C, H*W)
            corrcoeff = torch.corrcoef(image_flattened)

            for i, name in enumerate(channel_names):
                for j, other_name in enumerate(channel_names):
                    pairwise_correlations[(name, other_name)].append(corrcoeff[i, j].item())

        avg_pairwise_correlations = {pair: np.mean(corrs) for pair, corrs in pairwise_correlations.items()}
        results = pd.DataFrame([
            {'channel': pair[0], 'channel_2': pair[1], 'uniprot_id': channel_to_uniprot_map.get(pair[0], None), 'uniprot_id_2': channel_to_uniprot_map.get(pair[1], None), 'metric': 'avg_correlation', 'score': avg_corr} 
            for pair, avg_corr in avg_pairwise_correlations.items()
        ])
        results.to_parquet(results_dir / f'{dataset_name}_correlations.parquet')


if __name__ == "__main__":
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.load('configs/base_config.yaml')

    datasets_config = load_multiple_configs(cli_config.datasets_config)
    config = OmegaConf.merge(config, datasets_config, cli_config)

    logger.info(f'Config: \n{OmegaConf.to_yaml(config)}')

    set_seed(config.random_seed)
    compute_correlations(config)