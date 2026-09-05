import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict

from spora_bench.utils.setup_utils import load_multiple_configs, set_seed
from spora_io.datasets import MultiplexImagingDataset
from spora_bench.utils.instance_segmentation_utils import compute_matches
from sklearn.preprocessing import LabelEncoder
from spora_bench.utils.evaluation_utils import (bootstrap_classification_report,
    transform_bootstrap_report_to_df,
    transform_classification_report_to_df)
from sklearn.metrics import classification_report, confusion_matrix

from hydra.utils import instantiate

def run_cell_typing(
        config: DictConfig
    ):
    """
    Evaluates the model on instance segmentation benchmarks.
    Args:
        config: The configuration object containing model, training and dataset information.
    """
    output_dir = Path(config.output_dir)
    results_dir = output_dir / config.model.model_name / 'results'
    os.makedirs(results_dir, exist_ok=True)

    spora_model = instantiate(config.model)

    for dataset_key, dataset_config in config.datasets.items():
        logger.info(f'Running instance segmentation benchmarks for dataset: {dataset_key}')
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

        cell_metadata = pd.read_parquet(dataset.path / 'metadata' / 'cells.parquet')

        test_tissue_ids = dataset.tissue_modality_metadata[dataset.tissue_modality_metadata['split'] == 'test'].index.values

        cell_metadata = cell_metadata[cell_metadata['tissue_id'].isin(test_tissue_ids)]
        cell_metadata_per_tissue_id = {tissue_id: tissue_df.set_index('cell_id') for tissue_id, tissue_df in cell_metadata.groupby('tissue_id')}

        for task_name in dataset_config.benchmarks.cell_typing.keys():

            task_config = dataset_config.benchmarks.cell_typing[task_name]
            label_col = task_config.label_col
            class_order = task_config.class_order # list of classes
            label_encoder = LabelEncoder()
            label_encoder.classes_ = np.array(class_order)

            excluded_classes = task_config.excluded_classes
            excluded_classes_encoded = label_encoder.transform(excluded_classes).tolist()

            eval_classes = [c for c in class_order if c not in excluded_classes]

            logger.info(f'Running cell typing benchmark for label column: {label_col}')

            y_pred_all = []
            y_true_all = []

            for tissue_id in tqdm(test_tissue_ids, desc=f"Processing tissues for dataset {dataset_key}"):
                if not tissue_id in cell_metadata_per_tissue_id:
                    logger.warning(f"No cell metadata found for tissue {tissue_id}. Skipping this tissue.")
                    continue

                metadata = cell_metadata_per_tissue_id[tissue_id]

                tissue = dataset.get_tissue(tissue_id, kind="uniprot_filtered", preprocess=True, image_mode="CHW")
                true_cell_mask = dataset.get_cell_instance_mask(tissue_id,)

                if tissue.image.shape[1] < 256 or tissue.image.shape[2] < 256:
                    logger.warning(f"Tissue {tissue_id} has image size {tissue.image.shape[1:]} which is smaller than 256x256. Skipping this tissue.")
                    continue

                y_pred = spora_model.predict_cell_types(tissue, true_cell_mask, excluded_classes=excluded_classes_encoded)
                y_pred = y_pred.cpu().numpy() # length = number of cells in true_cell_mask

                cell_ids = np.unique(true_cell_mask.mask)
                cell_ids = cell_ids[cell_ids != 0] # exclude background
                assert len(cell_ids) == len(y_pred), f"Number of predicted cell types ({len(y_pred)}) does not match number of unique cells in mask ({len(cell_ids)}) for tissue {tissue_id}."

                y_pred_decoded = label_encoder.inverse_transform(y_pred)
                y_true_decoded = metadata.loc[cell_ids, label_col].values

                y_pred_all.append(y_pred_decoded)
                y_true_all.append(y_true_decoded)

            if not y_pred_all:
                logger.error(f"No tissues evaluated for {dataset_key}/{task_name}. Skipping results aggregation and saving..")
                continue

            y_pred_all = np.concatenate(y_pred_all)
            y_true_all = np.concatenate(y_true_all)

            logger.info('Evaluating results...')
            report = classification_report(y_true=y_true_all, y_pred=y_pred_all, labels=eval_classes, output_dict=True)
            report = transform_classification_report_to_df(report)
            report = report.assign(model=config.model.model_name, task=task_name)
            report.to_parquet(results_dir / f'{dataset_name}_{task_name}_classification_report.parquet')

            bootstrap_report = bootstrap_classification_report(y_true=y_true_all, y_pred=y_pred_all, n_bootstraps=1000, labels=eval_classes, random_state=42)
            bootstrap_report = transform_bootstrap_report_to_df(bootstrap_report, n_bootstraps=1000)
            bootstrap_report = bootstrap_report.assign(model=config.model.model_name, task=task_name)
            bootstrap_report.to_parquet(results_dir / f'{dataset_name}_{task_name}_bootstrap_classification_report.parquet')

            cm = confusion_matrix(y_true=y_true_all, y_pred=y_pred_all, labels=eval_classes)
            cm = pd.DataFrame(cm, index=eval_classes, columns=eval_classes)
            cm.to_parquet(results_dir / f'{dataset_name}_{task_name}_confusion_matrix.parquet')

            logger.info(f'Finished cell typing benchmark for dataset: {dataset_key}, task: {task_name}. Results saved to {results_dir}')



if __name__ == "__main__":
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.load('configs/base_config.yaml')

    model_config = OmegaConf.load(cli_config.model_config)

    datasets_config = load_multiple_configs(cli_config.datasets_config)
    benchmarks_config = load_multiple_configs(cli_config.benchmarks_config)

    config = OmegaConf.merge(config, model_config, datasets_config, benchmarks_config, cli_config)

    logger.info(f'Config: \n{OmegaConf.to_yaml(config)}')

    set_seed(config.random_seed)
    run_cell_typing(config)