import os
import pandas as pd
import numpy as np
import anndata as ad
import astir as ast

from omegaconf import OmegaConf, DictConfig
from loguru import logger
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch_scatter import scatter_mean
from tqdm import tqdm

from spora_bench.utils.cell_level_utils import config_has_cell_level_benchmark
from spora_bench.utils.setup_utils import set_seed, load_multiple_configs
from spora_bench.utils.evaluation_utils import bootstrap_classification_report, transform_bootstrap_report_to_df, transform_classification_report_to_df

from spora_io import MultiplexImagingDataset

def run_astir(config: DictConfig):
    """
    Runs astir for each dataset and cell-level benchmark task specified in the configuration, evaluates the results and saves them to disk.
    Args:
        config: DictConfig object containing the configuration for the datasets, benchmarks and model.
    """
    output_dir = Path(config.output_dir)
    results_dir = output_dir / config.model.model_name / 'results'
    intensities_dir = output_dir / config.model.model_name / 'mean_intensities'

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(intensities_dir, exist_ok=True)

    for dataset_key, dataset_config in config.datasets.items():
        if not config_has_cell_level_benchmark(dataset_config):
            logger.info(f"No cell-level benchmark found for dataset {dataset_key}. Skipping...")
            continue

        dataset_dir = Path(dataset_config.path)
        print(f'Running astir benchmark for dataset: {dataset_key} at {dataset_dir}')

        adata_path = intensities_dir / f'{dataset_key}.h5ad'
        if adata_path.exists():
            logger.info(f'Loading cell tokens for dataset {dataset_key} from existing AnnData file at {adata_path}...')
            adata = ad.read_h5ad(adata_path)
        else:
            logger.info(f'Computing mean intensities for dataset {dataset_key} for the first time and saving to {adata_path}...')
            adata = compute_astir_mean_intensities(dataset_config)
            adata.write_h5ad(adata_path)
        
        logger.info(f'Shape of AnnData object: {adata.shape}')

        # TODO load instead spora-io dataset 
        tissue_metadata = pd.read_parquet(dataset_dir / 'metadata' / 'tissues.parquet')
        cell_metadata = pd.read_parquet(dataset_dir / 'metadata' / 'cells.parquet')
        adata.obs = pd.merge(left=adata.obs, right=cell_metadata, on=['tissue_id', 'cell_id'], how='left', validate='one_to_one')

        # Create train/test split based on tissue_id, later we will need only test though but we include train here to determine complete list of classes.
        train_tids = tissue_metadata[tissue_metadata['split'] == 'train']['tissue_id'].unique()
        test_tids = tissue_metadata[tissue_metadata['split'] == 'test']['tissue_id'].unique()
        base_train_adata = adata[adata.obs['tissue_id'].isin(train_tids)]
        base_test_adata = adata[adata.obs['tissue_id'].isin(test_tids)]

        # Iterate over each cell-level benchmark task for the dataset and run Astir
        for task_name in dataset_config.benchmarks.cell_level:
            task_config = dataset_config.benchmarks.cell_level[task_name]
            label_col = task_config.label_col
            excluded_classes = task_config.excluded_classes
            logger.info(f'Running Astir for label column: {label_col} with excluded classes: {excluded_classes}')

            train_adata = base_train_adata.copy()
            train_adata = train_adata[~train_adata.obs[label_col].isna()]
            test_adata = base_test_adata.copy()
            test_adata = test_adata[~test_adata.obs[label_col].isna()]

            if excluded_classes:
                train_adata = train_adata[~train_adata.obs[label_col].isin(excluded_classes)]
                test_adata = test_adata[~test_adata.obs[label_col].isin(excluded_classes)]
            
            # Forbid Unknown and Other classes as these have a special meaning in astir.
            unique_classes = np.unique(np.concatenate([train_adata.obs[label_col].unique(), test_adata.obs[label_col].unique()]))
            if "Other" in unique_classes:
                raise ValueError(f"The 'Other' class is not allowed in the label column for Astir benchmarks as it has a special meaning in astir.")
            if "Unknown" in unique_classes:
                raise ValueError(f"The 'Unknown' class is not allowed in the label column for Astir benchmarks as it has a special meaning in astir.")
            
            # Check that configuration fits the underlying data (e.g. all required markers are present in the adata, all cell types in the marker config match cell types in the data.) 
            marker_config = OmegaConf.create({"cell_types": task_config['cell_types']})
            cell_types = marker_config.cell_types.keys()
            if not set(cell_types) == set(unique_classes):
                raise ValueError(f"Marker config cell types {cell_types} do not match the unique classes in the data {unique_classes}. Please check your benchmark configuration for task {task_name} in dataset {dataset_key}.")
            
            required_markers = []
            for _, markers in marker_config.cell_types.items():
                required_markers.extend(markers)
            required_markers = list(set(required_markers))
            if not set(required_markers).issubset(set(adata.var_names)):
                missing_markers = set(required_markers) - set(adata.var_names)
                raise ValueError(f"The following required markers for Astir are missing from the adata: {missing_markers}. Please check dataset column names and the benchmark configuration for task {task_name} in dataset {dataset_key}.")
            
            test_adata = test_adata[:, required_markers]
            if pd.isna(test_adata.X).any():
                raise ValueError(
                    f"NaN values found in the input data for Astir in dataset {dataset_key}. Please check the features in `adata` for NaN values in the required markers.\n"
                    "Astir requires all listed markers to be present. If a cohort uses varying panels across samples, only the markers common to all samples should be included in the marker configuration for the benchmark."
                )
            X = pd.DataFrame(test_adata.X, columns=test_adata.var_names, index=test_adata.obs_names)
            X = np.arcsinh(X / config.model.arcsinh_cofactor) 
            y_true = test_adata.obs[label_col].values

            # Run astir
            logger.info(f"Fitting Astir model...")
            astir_model = ast.Astir(input_expr=X, marker_dict=marker_config)
            astir_model.fit_type(max_epochs=config.model.max_epochs, n_init=config.model.n_init, n_init_epochs=config.model.n_init_epochs)

            logger.info(f"Making predictions...")
            y_pred = astir_model.get_celltypes(threshold=config.model.threshold).cell_type

            logger.info(f"Evaluating results...")

            # Evaluate and save results
            logger.info('Evaluating results...')
            report = classification_report(y_true=y_true, y_pred=y_pred, labels=unique_classes, output_dict=True)
            report = transform_classification_report_to_df(report)
            report = report.assign(model=config.model.model_name, task=label_col)
            report.to_parquet(results_dir / f'{dataset_key}_{label_col}_classification_report.parquet')

            # CAREFUL: for astir we can't use the fast bootstrap implementation at the moment as it behaves differently in the special case that labels is a subset of all occuring labels
            # TODO make this consistent but still compatible with astir's special classes Unknown and Other
            bootstrap_report = bootstrap_classification_report(y_true=y_true, y_pred=y_pred, n_bootstraps=1000, labels=unique_classes, random_state=42)
            bootstrap_report = transform_bootstrap_report_to_df(bootstrap_report, n_bootstraps=1000)
            bootstrap_report = bootstrap_report.assign(model=config.model.model_name, task=label_col)
            bootstrap_report.to_parquet(results_dir / f'{dataset_key}_{label_col}_bootstrap_classification_report.parquet')

            # We remove the computation of the confusion matrix as as we should not disregard here the special classes Unknown and Other. Otherwise computing the recall-normalized confusion matrix later will diverge from the classification report.
            # We keep the code here for now but it is commented out. If you want to compute a confusion matrix for astir results, please make sure to interpret it correctly in the context of astir's special classes.
            # cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=unique_classes)
            # cm = pd.DataFrame(cm, index=unique_classes, columns=unique_classes)
            # cm.to_parquet(results_dir / f'{dataset_key}_{label_col}_confusion_matrix.parquet')

            logger.info(f'Finished Astir benchmark for dataset {dataset_key} and task {label_col}. Results saved to {results_dir}')



def compute_astir_mean_intensities(dataset_config: DictConfig) -> ad.AnnData:
    """
    Compute mean intensities for astir. Only quality control channels common to all images in the dataset are included.
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
    adata = ad.AnnData(X=mean_intensities, var=pd.DataFrame(index=common_qc_channels), obs=pd.DataFrame({'tissue_id': tissue_ids, 'cell_id': cell_ids}))       
    return adata


if __name__ == "__main__":
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.load('configs/base_config.yaml')
    model_config = OmegaConf.load(f'configs/models/astir.yaml')

    datasets_config = load_multiple_configs(cli_config.datasets_config)
    benchmarks_config = load_multiple_configs(cli_config.benchmarks_config)

    config = OmegaConf.merge(config, model_config, datasets_config, benchmarks_config, cli_config)

    set_seed(config.random_seed)
    run_astir(config)


