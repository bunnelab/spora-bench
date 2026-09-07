import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
from anndata import read_h5ad
from loguru import logger
from omegaconf import OmegaConf, DictConfig
import torch
from tqdm import tqdm
from torch_scatter import scatter_mean
import anndata as ad
import datetime

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from spora_io.datasets import MultiplexImagingDataset
from spora_bench.utils.setup_utils import load_multiple_configs, set_seed
from spora_bench.utils.evaluation_utils import (bootstrap_classification_report,
    transform_bootstrap_report_to_df,
    transform_classification_report_to_df)
from spora_bench.utils.intensities_utils import compute_mean_intensities

from maps.cell_phenotyping import Trainer, Predictor


def run_maps(config: DictConfig):
    """
    Runs MAPS.
    Args:
        config (DictConfig): The configuration object containing model, datasets and benchmark information.
    """    
    output_dir = Path(config.output_dir)
    results_dir = output_dir / config.model.model_name / 'results'
    intensities_dir = output_dir / config.model.model_name / 'normalized_mean_intensities'

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(intensities_dir, exist_ok=True)

    train_adatas = {}
    val_adatas = {}
    test_adatas = {}

    for train_dataset_key, train_dataset_config in config.train_datasets.items():
        logger.info(f'Processing train dataset: {train_dataset_key}')

        dataset_dir = Path(train_dataset_config.path)

        adata_path = intensities_dir / f"{train_dataset_key}.h5ad"
        if adata_path.exists():
            logger.info(f"Loading precomputed normalized mean intensities for dataset {train_dataset_key} from {adata_path}")
            adata = read_h5ad(adata_path)
        else:
            logger.info(f"Computing normalized mean intensities for dataset {train_dataset_key}")
            adata = compute_mean_intensities(train_dataset_config, normalize=True)
            adata.write_h5ad(adata_path)

        logger.info(f'Shape of AnnData object for dataset {train_dataset_key}: {adata.shape}')

        tissue_metadata = pd.read_parquet(dataset_dir / 'metadata' / 'tissues.parquet')
        cell_metadata = pd.read_parquet(dataset_dir / 'metadata' / 'cells.parquet')
        cell_metadata.cell_id = cell_metadata.cell_id.astype(int)  # ensure cell_id is int for merging

        adata.obs = pd.merge(left=adata.obs, right=cell_metadata, on=['tissue_id', 'cell_id'], how='left', validate='one_to_one')

        # all modalities but does not matter
        train_tids = tissue_metadata[tissue_metadata.split == 'train'].tissue_id.unique()
        adata = adata[adata.obs['tissue_id'].isin(train_tids)]
        tids = adata.obs['tissue_id'].unique()
        pids = [tid.split("_")[1] for tid in tids] 

        gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
        train_idx, val_idx = next(gss.split(tids, groups=pids))
        assert len(val_idx) > 0, "Validation set is empty. Please check the dataset and grouping strategy."

        train_tids = tids[train_idx]
        val_tids = tids[val_idx]

        train_adata = adata[adata.obs['tissue_id'].isin(train_tids)].copy()
        val_adata = adata[adata.obs['tissue_id'].isin(val_tids)].copy()

        train_adatas[train_dataset_key] = train_adata
        val_adatas[train_dataset_key] = val_adata

    for test_dataset_key, test_dataset_config in config.test_datasets.items():
        logger.info(f'Processing test dataset: {test_dataset_key}')

        dataset_dir = Path(test_dataset_config.path)

        adata_path = intensities_dir / f"{test_dataset_key}.h5ad"
        if adata_path.exists():
            logger.info(f"Loading precomputed normalized mean intensities for dataset {test_dataset_key} from {adata_path}")
            adata = read_h5ad(adata_path)
        else:
            logger.info(f"Computing normalized mean intensities for dataset {test_dataset_key}")
            adata = compute_mean_intensities(test_dataset_config, normalize=True)
            adata.write_h5ad(adata_path)

        logger.info(f'Shape of AnnData object for dataset {test_dataset_key}: {adata.shape}')

        tissue_metadata = pd.read_parquet(dataset_dir / 'metadata' / 'tissues.parquet')
        cell_metadata = pd.read_parquet(dataset_dir / 'metadata' / 'cells.parquet')
        cell_metadata.cell_id = cell_metadata.cell_id.astype(int)  # ensure cell_id is int for merging

        adata.obs = pd.merge(left=adata.obs, right=cell_metadata, on=['tissue_id', 'cell_id'], how='left', validate='one_to_one')

        # all modalities but does not matter
        test_tids = tissue_metadata[tissue_metadata.split == 'test'].tissue_id.unique()
        test_adata = adata[adata.obs['tissue_id'].isin(test_tids)].copy()
        test_adatas[test_dataset_key] = test_adata


    # We match features across datasets by their uniprot ids. Channel names for the same uniprot id may differ across datasets.
    common_uniprots = set.intersection(*[set(adata.var['uniprot_id']) for adata in list(train_adatas.values()) + list(test_adatas.values())])
    logger.info(f"Found {len(common_uniprots)} common uniprots across all datasets: {common_uniprots}")
    # filter for uniprots which match uniprot format
    uniprot_regex = r'^[OPQ][0-9][A-Z0-9]{3}[0-9]|^[A-NR-Z][0-9][A-Z0-9]{3}[0-9]'
    print(uniprot_regex)
    common_uniprots = sorted([uniprot for uniprot in common_uniprots if not pd.isna(uniprot) and re.match(uniprot_regex, uniprot)])
    logger.info(f"Found {len(common_uniprots)} common uniprots across all datasets: {common_uniprots}")

    def _build_task_df(adata, label_col, common_uniprots, included_classes):
        adata = adata[(~adata.obs[label_col].isna()) & (adata.obs[label_col].isin(included_classes))].copy()
        uniprot_to_channel = dict(zip(adata.var['uniprot_id'], adata.var.index))
        channels = [uniprot_to_channel[u] for u in common_uniprots]
        df = pd.DataFrame(adata[:, channels].X, columns=common_uniprots)
        df['cell_label'] = adata.obs[label_col].values
        return df

    for test_dataset_key, test_dataset_config in config.test_datasets.items():
        test_dataset_name = test_dataset_config.name
        logger.info(f'Running MAPS benchmarks for test dataset: {test_dataset_name}')

        for task_name, task_config in test_dataset_config.benchmarks.cell_annotation.items():
            label_col = task_config.label_col
            included_classes = list(task_config.included_classes)
            logger.info(f'Running MAPS transfer benchmark for test dataset: {test_dataset_name}, task: {task_name}, label column: {label_col}, classes: {included_classes}')

            test_adata = test_adatas[test_dataset_key]
            test_df = _build_task_df(test_adata, label_col, common_uniprots, included_classes)

            to_concat_train = []
            to_concat_val = []

            for train_dataset_key, train_adata in train_adatas.items():
                train_df = _build_task_df(train_adata, label_col, common_uniprots, included_classes)
                to_concat_train.append(train_df)

            for val_dataset_key, val_adata in val_adatas.items():
                val_df = _build_task_df(val_adata, label_col, common_uniprots, included_classes)
                to_concat_val.append(val_df)

            train_df = pd.concat(to_concat_train, axis=0, ignore_index=True)
            val_df = pd.concat(to_concat_val, axis=0, ignore_index=True)

            num_markers = len(common_uniprots)
            occurring_classes = np.unique(np.concatenate([
                                                    train_df.cell_label.unique(),
                                                    val_df.cell_label.unique(),
                                                    test_df.cell_label.unique()
                                                    ]))

            assert set(occurring_classes).issubset(set(included_classes)), f"Found unexpected classes in the data. Occurring classes: {occurring_classes}, expected classes: {included_classes}"

            num_occurring_classes = len(occurring_classes)

            # get timestamp to create a unique temporary directory for this run
            tmp_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{test_dataset_name}_{task_name}"
            tmp_dir = results_dir / "tmp" / tmp_tag

            os.makedirs(tmp_dir, exist_ok=True)

            label_encoder = LabelEncoder()
            label_encoder.fit(occurring_classes)

            train_df['cell_label'] = label_encoder.transform(train_df['cell_label'])
            val_df['cell_label'] = label_encoder.transform(val_df['cell_label'])
            test_df['cell_label'] = label_encoder.transform(test_df['cell_label'])

            train_df.to_csv(tmp_dir / "train.csv", index=False)
            val_df.to_csv(tmp_dir / "val.csv", index=False)
            test_df.to_csv(tmp_dir / "test.csv", index=False)

            logger.info(f'Training MAPS model on {len(train_df)} samples, validating on {len(val_df)} samples, and testing on {len(test_df)} samples for dataset: {test_dataset_name} and task: {task_name}.')

            trainer = Trainer(results_dir=tmp_dir,
                            num_features=num_markers,
                            num_classes=num_occurring_classes,
                            batch_size=128,
                            max_epochs=300,
                            min_epochs=100,
                            patience=20,
                            verbose=0
                        )

            trainer.fit(tmp_dir / "train.csv", tmp_dir / "val.csv")

            model_path = tmp_dir / "best_checkpoint.pt"

            model = Predictor(model_checkpoint_path=model_path, num_features=num_markers, num_classes=num_occurring_classes, batch_size=128)
            pred_labels, pred_probs = model.predict(tmp_dir / "test.csv")

            y_true = label_encoder.inverse_transform(test_df.cell_label.values)
            y_pred = label_encoder.inverse_transform(pred_labels)

            logger.info('Evaluating results...')
            report = classification_report(y_true, y_pred, labels=included_classes, output_dict=True)
            report_df = transform_classification_report_to_df(report)
            report_df = report_df.assign(model=config.model.model_name, task=task_name)
            report_df.to_parquet(results_dir / f'{test_dataset_name}_{task_name}_classification_report.parquet')

            bootstrap_report = bootstrap_classification_report(y_true, y_pred, n_bootstraps=1000, labels=included_classes, random_state=42)
            bootstrap_report_df = transform_bootstrap_report_to_df(bootstrap_report, n_bootstraps=1000)
            bootstrap_report_df = bootstrap_report_df.assign(model=config.model.model_name, task=task_name)
            bootstrap_report_df.to_parquet(results_dir / f'{test_dataset_name}_{task_name}_bootstrap_classification_report.parquet')
            
            cm = confusion_matrix(y_true, y_pred, labels=included_classes)
            cm = pd.DataFrame(cm, index=included_classes, columns=included_classes)
            cm.to_parquet(results_dir / f'{test_dataset_name}_{task_name}_confusion_matrix.parquet')

            logger.info(f'Finished MAPS benchmark for {test_dataset_name} with task {task_name}. Results saved to {results_dir}')


if __name__ == "__main__":
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.load('configs/base_config.yaml')

    model_config = OmegaConf.load(cli_config.model_config)

    train_datasets_config = load_multiple_configs(cli_config.train_datasets_config)
    test_datasets_config = load_multiple_configs(cli_config.test_datasets_config)

    train_datasets_config.train_datasets = train_datasets_config.datasets
    test_datasets_config.test_datasets = test_datasets_config.datasets
    del train_datasets_config.datasets
    del test_datasets_config.datasets

    benchmarks_config = load_multiple_configs(cli_config.benchmarks_config)
    benchmarks_config.test_datasets = benchmarks_config.datasets
    del benchmarks_config.datasets

    config = OmegaConf.merge(config, model_config, train_datasets_config, test_datasets_config, benchmarks_config, cli_config)

    set_seed(config.random_seed)
    run_maps(config)
