from pathlib import Path

import anndata as ad
import numpy as np
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from spora_io import MultiplexImagingDataset
from tqdm import tqdm


def compute_cell_tokens(config: OmegaConf):
    """
    Compute cell tokens for each dataset in the configuration for the specified model. The computed cell tokens are saved in an AnnData format (.h5ad) for each dataset.
    Args:
        config (OmegaConf): The configuration object containing model and downstream dataset information.
    """

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

        all_cell_ids = []
        all_cell_tokens = []
        all_tissue_ids = []

        for tissue_id in tqdm(dataset.get_tissue_ids(kind="modality")):
            cell_ids, cell_tokens = spora_model.compute_cell_tokens(dataset, tissue_id) 

            all_cell_ids.append(cell_ids)
            all_cell_tokens.append(cell_tokens)
            all_tissue_ids.extend([tissue_id] * len(cell_ids))

        all_cell_ids = torch.cat(all_cell_ids, dim=0).numpy()
        all_cell_tokens = torch.cat(all_cell_tokens, dim=0).numpy()
        all_tissue_ids = np.array(all_tissue_ids)

        adata = ad.AnnData(X=all_cell_tokens)
        adata.obs['cell_id'] = all_cell_ids
        adata.obs['tissue_id'] = all_tissue_ids

        tgt_dir = Path(f'{config.output_dir}/{config.model.model_name}/cell_tokens')
        tgt_dir.mkdir(parents=True, exist_ok=True)
        tgt_path = tgt_dir / f'{dataset_name}.h5ad'
        adata.write_h5ad(tgt_path)
        logger.info(f'Saved cell tokens for dataset {dataset_name} to {tgt_path}')


if __name__ == "__main__":
    cli_config = OmegaConf.from_cli()

    base_config = OmegaConf.load('configs/base_config.yaml')
    model_config = OmegaConf.load(cli_config.model_config)
    datasets_config = OmegaConf.load(cli_config.datasets_config)

    config = OmegaConf.merge(base_config, model_config, datasets_config, cli_config)
    logger.info(f'Merged config:\n{OmegaConf.to_yaml(config)}')

    compute_cell_tokens(config)