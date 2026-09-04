import os
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from hydra.utils import instantiate
from collections import defaultdict

import torch

from spora_bench.utils.setup_utils import load_multiple_configs, set_seed
from spora_io.datasets import MultiplexImagingDataset, MultiplexTissue
from spora_bench.wrapper import MarkerNotSupportedError
from tqdm import tqdm

def run_virtual_stainings(config): 

    output_dir = Path(config.output_dir)
    results_dir = output_dir / config.model.model_name / 'results'
    os.makedirs(results_dir, exist_ok=True)

    spora_model = instantiate(config.model)
    
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
        
        correlations = defaultdict(list)
        mses = defaultdict(list)
        channel_to_uniprot_map = dict(zip(dataset.channel_list['channel_name'], dataset.channel_list['uniprot_id']))

        test_tissue_ids = dataset.tissue_modality_metadata[dataset.tissue_modality_metadata['split'] == 'test'].index.values

        for tissue_id in tqdm(test_tissue_ids, desc=f"Processing tissues for dataset {dataset_key}"):

            tissue = dataset.get_tissue(tissue_id, kind="uniprot_filtered", preprocess=True, image_mode="CHW")
            
            if tissue.image.shape[1] < 256 or tissue.image.shape[2] < 256:
                logger.warning(f"Tissue {tissue_id} has image size {tissue.image.shape[1:]} which is smaller than 256x256. Skipping this tissue.")
                continue

            logger.info(f"Processing tissue: {tissue_id} with {len(tissue.channel_names)} channels.")
            
            if getattr(config, "rescale", False):
                x = tissue.image # (C, H, W)
                max_channel_values = x.reshape(x.shape[0], -1).max(dim=1).values
                min_channel_values = x.reshape(x.shape[0], -1).min(dim=1).values
                x = (x - min_channel_values[:, None, None]) / (max_channel_values[:, None, None] - min_channel_values[:, None, None])
                tissue.image = x
            
            
            for cidx in range(len(tissue.channel_names)):
                channel_to_predict = tissue.channel_names[cidx]
                uniprot_to_predict = tissue.uniprot_ids[cidx]
                channel_true = tissue.image[cidx]
                
                bmask_to_drop = (tissue.channel_names == channel_to_predict)
                if bmask_to_drop.sum() > 1:
                    raise ValueError(f"Channel {channel_to_predict} appears multiple times in tissue {tissue_id}. All channel names within a dataset should be unique. Cannot perform virtual staining benchmark.")
                bmask_to_drop_mapped_to_image_loading_mask = np.zeros_like(tissue.image_loading_mask, dtype=bool)
                bmask_to_drop_mapped_to_image_loading_mask[tissue.image_loading_mask] = bmask_to_drop

                tissue_without_channel = MultiplexTissue(
                    tissue_id=tissue.tissue_id,
                    image=tissue.image[~bmask_to_drop],
                    channel_names=tissue.channel_names[~bmask_to_drop],
                    uniprot_ids=tissue.uniprot_ids[~bmask_to_drop],
                    measured_mask=tissue.measured_mask,
                    image_loading_mask=tissue.image_loading_mask & ~bmask_to_drop_mapped_to_image_loading_mask,
                )

                try:
                    virtual_stain = spora_model.predict_marker(tissue_without_channel, target_channel_name=channel_to_predict, target_uniprot_id=uniprot_to_predict)
                except MarkerNotSupportedError as e:
                    logger.warning(f"Skipping tissue {tissue_id} for channel {channel_to_predict} due to error: {e}")
                    continue
                    
                corr = np.corrcoef(channel_true.flatten(), virtual_stain.flatten())[0, 1].item()
                # Correlation can be nan if the true channel is constant (e.g., due to clipping).
                if np.isnan(corr):
                    logger.warning(f"Correlation is NaN for tissue {tissue_id} and channel {channel_to_predict}. Skipping this tissue-channel pair.")
                else:
                    correlations[channel_to_predict].append(corr)
                mses[channel_to_predict].append(((channel_true.flatten() - virtual_stain.flatten())**2).mean().item())
                
                
        avg_correlations = {channel: np.mean(corrs) for channel, corrs in correlations.items()}
        avg_mses = {channel: np.mean(mse_list) for channel, mse_list in mses.items()}
        results = []
        for channel in correlations.keys():
            results.append({
                'dataset': dataset_name,
                'channel': channel,
                'uniprot_id': channel_to_uniprot_map.get(channel, None),
                'metric': 'avg_correlation',
                'score': avg_correlations[channel],
            })
            results.append({
                'dataset': dataset_name,
                'channel': channel,
                'uniprot_id': channel_to_uniprot_map.get(channel, None),
                'metric': 'avg_mse',
                'score': avg_mses[channel],
            })
        results = pd.DataFrame(results)
        results = results.assign(model=config.model.model_name)
        results.to_parquet(results_dir / f'{dataset_name}_virtual_staining_results.parquet')
        logger.info(f'Finished virtual staining benchmark for dataset {dataset_name}. Results saved to {results_dir}')



if __name__ == "__main__":
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.load('configs/base_config.yaml')

    model_config = OmegaConf.load(cli_config.model_config)

    datasets_config = load_multiple_configs(cli_config.datasets_config)
    config = OmegaConf.merge(config, model_config, datasets_config, cli_config)

    logger.info(f'Config: \n{OmegaConf.to_yaml(config)}')

    set_seed(config.random_seed)
    run_virtual_stainings(config)