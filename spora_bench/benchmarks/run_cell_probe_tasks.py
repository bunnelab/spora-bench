import os
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import read_h5ad

try:
    from cuml.linear_model import LogisticRegression
except Exception as e:
    print("Could not import cuML. Make sure you have cuML installed and a compatible GPU. Falling back to CPU implementation.")
    from sklearn.linear_model import LogisticRegression

from loguru import logger
from omegaconf import OmegaConf, DictConfig
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from spora_bench.utils.evaluation_utils import (bootstrap_classification_report,
    transform_bootstrap_report_to_df,
    transform_classification_report_to_df)
from spora_bench.utils.setup_utils import load_multiple_configs, set_seed
from spora_bench.tools.compute_cell_tokens import compute_cell_tokens
from spora_bench.utils.evaluation_utils import (
    transform_bootstrap_report_to_df, transform_classification_report_to_df)
from spora_bench.utils.setup_utils import load_multiple_configs, set_seed
from spora_bench.utils.tissue_level_utils import get_patient_splits


def run_linear_probes(
        config: DictConfig
        ):
    """Evaluate the model on cell-level benchmarks using linear probes. If cell tokens have been computed previously, we load them from this location: {config.output_dir}/{model_name}/cell_tokens/{dataset_key}.h5ad.
    Args: 
        config: The configuration object containing model, dataset and benchmark information.
    """
    output_dir = Path(config.output_dir)
    results_dir = output_dir / config.model.model_name / 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    for dataset_key, dataset_config in config.datasets.items():
        if not "benchmarks" in dataset_config or not "cell_level" in dataset_config.benchmarks:
            logger.info(f"No cell-level benchmark found for dataset {dataset_key}. Skipping...")
            continue

        dataset_name = dataset_config.name
        dataset_dir = Path(dataset_config.path)
        logger.info(f'Running benchmarks for dataset: {dataset_key} at {dataset_dir}')
    
        adata_path = output_dir / config.model.model_name / 'cell_tokens' / f'{dataset_key}.h5ad'
        if not adata_path.exists():
            logger.info(f'Cell tokens file not found at {adata_path}. Computing cell tokens for {dataset_key}...')
            single_dataset_config = config.copy()
            single_dataset_config.datasets = {dataset_key: dataset_config}
            compute_cell_tokens(single_dataset_config)

        logger.info(f'Loading cell tokens for dataset {dataset_key}...')
        adata = read_h5ad(adata_path)
        logger.info(f'Shape of loaded cell tokens adata: {adata.shape}')

        # TODO load instead spora-io dataset 
        cell_metadata = pd.read_parquet(dataset_dir / 'metadata' / 'cells.parquet')
        adata.obs = pd.merge(left=adata.obs, right=cell_metadata, on=['tissue_id', 'cell_id'], how='left', validate='one_to_one')

        # Load train test splits based on patient splits provided by spora-data
        patient_splits = get_patient_splits(dataset_config)
        all_tids = adata.obs["tissue_id"].unique()
        train_tids = [tid for tid in all_tids if patient_splits[tid.split("_")[1]] == "train"]
        test_tids = [tid for tid in all_tids if patient_splits[tid.split("_")[1]] == "test"]

        base_test_adata = adata[adata.obs['tissue_id'].isin(test_tids)]
        base_train_adata = adata[adata.obs['tissue_id'].isin(train_tids)]

        # TODO clean up this task_config.task structure
        for task_name in dataset_config.benchmarks.cell_level.keys():
            task_config = dataset_config.benchmarks.cell_level[task_name]
            label_col = task_config.label_col
            excluded_classes = task_config.excluded_classes
            logger.info(f'Running cell level benchmark for label column: {label_col} with excluded classes: {excluded_classes}')

            # apply NA filter and exclude classes filter
            train_adata = base_train_adata.copy() # copy to avoid modifying the original adata which is used for multiple tasks
            test_adata = base_test_adata.copy() 
            train_adata = train_adata[~train_adata.obs[label_col].isna()]
            test_adata = test_adata[~test_adata.obs[label_col].isna()]

            if excluded_classes:
                train_adata = train_adata[~train_adata.obs[label_col].isin(excluded_classes)]
                test_adata = test_adata[~test_adata.obs[label_col].isin(excluded_classes)]

            scaler = StandardScaler()
            train_X = train_adata.X
            train_X = scaler.fit_transform(train_X) # standardize features
            test_X = test_adata.X
            test_X = scaler.transform(test_X)
            train_labels = train_adata.obs[label_col].values
            test_labels = test_adata.obs[label_col].values

            logger.info(f'Number training samples: {train_X.shape[0]}')
            logger.info(f'Number test samples: {test_X.shape[0]}')

            unique_classes = np.unique(np.concatenate([train_labels, test_labels]))
            le = LabelEncoder()
            le.fit(unique_classes)

            train_y = le.transform(train_labels)

            linear_model = LogisticRegression()

            logger.info('Fitting logistic regression model...')
            linear_model.fit(train_X, train_y)

            logger.info('Making predictions...')
            test_y_pred = linear_model.predict(test_X)
            test_y_pred_decoded = le.inverse_transform(test_y_pred)

            # Calculate and save evaluation metrics
            logger.info('Evaluating results...')
            report = classification_report(y_true=test_labels, y_pred=test_y_pred_decoded, labels=unique_classes, output_dict=True)
            report = transform_classification_report_to_df(report)
            report = report.assign(model=config.model.model_name, task=task_name)
            report.to_parquet(results_dir / f'{dataset_name}_{task_name}_classification_report.parquet')

            bootstrap_report = bootstrap_classification_report(y_true=test_labels, y_pred=test_y_pred_decoded, n_bootstraps=1000, labels=unique_classes, random_state=42)
            bootstrap_report = transform_bootstrap_report_to_df(bootstrap_report, n_bootstraps=1000)
            bootstrap_report = bootstrap_report.assign(model=config.model.model_name, task=task_name)
            bootstrap_report.to_parquet(results_dir / f'{dataset_name}_{task_name}_bootstrap_classification_report.parquet')

            cm = confusion_matrix(y_true=test_labels, y_pred=test_y_pred_decoded, labels=unique_classes)
            cm = pd.DataFrame(cm, index=unique_classes, columns=unique_classes)
            cm.to_parquet(results_dir / f'{dataset_name}_{task_name}_confusion_matrix.parquet')

            logger.info(f'Finished cell-level benchmark for dataset {dataset_name}, task {task_name} and label column {label_col}. Results saved to {results_dir}')


if __name__ == "__main__":
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.load('configs/base_config.yaml')

    model_config = OmegaConf.load(cli_config.model_config)

    datasets_config = load_multiple_configs(cli_config.datasets_config)
    benchmarks_config = load_multiple_configs(cli_config.benchmarks_config)

    config = OmegaConf.merge(config, model_config, datasets_config, benchmarks_config, cli_config)

    logger.info(f'Config: \n{OmegaConf.to_yaml(config)}')

    set_seed(config.random_seed)
    run_linear_probes(config)