
import os
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf, DictConfig
import torch
from tqdm import tqdm
from torch_scatter import scatter_mean
import anndata as ad
from spora_io.datasets import MultiplexImagingDataset


def compute_mean_intensities(dataset_config: DictConfig, normalize=False) -> ad.AnnData:
    """
    Compute mean intensities for astir. Only quality controlled channels common to all images in the dataset are included.
    Args:
        dataset_config: DictConfig object containing the dataset configuration.
    Returns:
        ad.AnnData: AnnData object with mean intensities, `var_names` equal to `channel_names` and obs containing `tissue_id` and `cell_id`.
    """
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

    tissue_ids = []
    cell_ids = []
    mean_intensities = []

    channels_per_tissue = dataset.image_channel_map
    common_channels_mask = channels_per_tissue.all(axis=0).values
    qc_mask = dataset.channel_list['qc_pass'].values
    common_qc_channels_mask = common_channels_mask & qc_mask
    common_qc_channels = dataset.channel_list['channel_name'][common_qc_channels_mask].values
    common_qc_uniprot_ids = dataset.channel_list['uniprot_id'][common_qc_channels_mask].values

    for tissue_id in tqdm(dataset.get_tissue_ids(kind="modality")):
        tissue = dataset.get_tissue(tissue_id, kind='qc_filtered', preprocess=False, image_mode="CHW")
        channel_names = tissue.channel_names
        channel_mask = np.isin(channel_names, common_qc_channels)
        if not np.array_equal(channel_names[channel_mask], common_qc_channels):
            raise ValueError(f"Channel names for tissue {tissue_id} do not match the common QC-passed channel names across the dataset. Please check the dataset and benchmark configuration. Tissue channel names: {channel_names[channel_mask]}, common QC-passed channel names: {common_qc_channels}.")
        
        tissue_image = tissue.image[channel_mask]
        C, H, W = tissue_image.shape
        try:
            segmentation_mask = dataset.get_cell_instance_mask(tissue_id)
            segmentation_mask = segmentation_mask.mask.int()
        except ValueError:
            logger.warning(f"Cell instance mask not found for tissue ID {tissue_id}. Skipping cell token computation.")
            continue
    
        means = scatter_mean(tissue_image.reshape(C, -1).T, segmentation_mask.reshape(-1), dim=0) # (max_cell_id+1, C)
        for cell_id in torch.unique(segmentation_mask):
            if cell_id == 0:  # skip background
                continue
            cell_mean_intensity = means[cell_id]
            tissue_ids.append(tissue_id)
            cell_ids.append(cell_id.item())
            mean_intensities.append(cell_mean_intensity)

    if len(mean_intensities) == 0:
        raise ValueError("No valid cells found across tissues.")

    mean_intensities = torch.stack(mean_intensities).numpy()

    if normalize:
        mean_intensities = (mean_intensities - mean_intensities.mean(axis=0)) / mean_intensities.std(axis=0).clip(min=1e-8)  # z-score normalization

    var = pd.DataFrame(index=common_qc_channels, data={'uniprot_id': common_qc_uniprot_ids})
    obs = pd.DataFrame({'tissue_id': tissue_ids, 'cell_id': cell_ids})
    adata = ad.AnnData(X=mean_intensities, var=var, obs=obs)       
    return adata