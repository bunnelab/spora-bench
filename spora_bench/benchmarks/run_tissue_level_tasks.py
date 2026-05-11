import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from spora_bench.tools.compute_tile_tokens import (compute_tile_tokens,
                                                   load_tile_tokens)
from spora_bench.utils.evaluation_utils import (
    bootstrap_classification_report, transform_bootstrap_report_to_df,
    transform_classification_report_to_df)
from spora_bench.utils.setup_utils import load_multiple_configs, set_seed
from spora_bench.utils.tissue_level_utils import (
    abmil_train_test, config_has_tissue_level_benchmark, get_patient_splits,
    get_targets)


def run_spora_split_validation(
        config: OmegaConf
        ):
    """Evaluate the model on the spora-io train-test splits for each dataset that contains tissue-level benchmarks. If tile tokens have been computed previously, we load them from this location: {config.output_dir}/{model_name}/tile_tokens/{dataset_key}.
    Args: 
        config: The configuration object containing model, training and dataset information.
    """
    output_dir = Path(config.output_dir)
    results_dir = output_dir / config.model.model_name / 'results'
    os.makedirs(results_dir, exist_ok=True)

    for dataset_key, dataset_config in config.datasets.items():
        if not config_has_tissue_level_benchmark(dataset_config):
            logger.info(f"Skipping dataset '{dataset_key}' as it does not have a tissue-level benchmark...")
            continue
        
        dataset_name = dataset_config.name
        tokens_dir = output_dir / config.model.model_name / "tile_tokens" / dataset_key

        if not tokens_dir.exists():
            logger.info(f"Computing tile tokens for dataset '{dataset_key}'...")
            single_dataset_config = config.copy()
            single_dataset_config.datasets = {dataset_key: dataset_config}
            compute_tile_tokens(single_dataset_config)

        tile_tokens = load_tile_tokens(config.output_dir, dataset_key, config.model.model_name)

        # Calculate tile tokens for patient-level tasks (X)
        logger.info(f"Post-Processing tile tokens for patient-level tasks for dataset '{dataset_key}'...")
        spora_model = instantiate(config.model)
        n_tokens_per_tissue = {tid: tokens.shape[0] for tid, tokens in tile_tokens.items()}
        tile_tokens_processed = torch.concat(list(tile_tokens.values()))
        tile_tokens_processed = spora_model.postprocess_tile_embeddings(tile_tokens_processed)
        tile_tokens_processed = torch.split(tile_tokens_processed, list(n_tokens_per_tissue.values()), dim=0)
        tile_tokens = {tid: tile_tokens_processed[idx] for idx, tid in enumerate(tile_tokens.keys())}

        # Prepare targets for patient-level tasks (y)
        targets = get_targets(dataset_config)
        
        # Get spora-io train-test splits for patient-level tasks
        patient_split_dict = get_patient_splits(dataset_config)

        for label_col, tissue_labels in targets.items():
            label_encoder = LabelEncoder()
            label_encoder.fit(list(tissue_labels.values()))
            tissue_labels_encoded = {
                tid: int(encoded_label)
                for tid, encoded_label in zip(
                    tissue_labels.keys(),
                    label_encoder.transform(list(tissue_labels.values())),
                )
            }

            # only use valid patient IDs (those with both tile tokens and labels)
            patient_ids = np.array([tid.split("_")[1] for tid in tissue_labels.keys() if tid in tile_tokens])
            patient_ids = np.unique(patient_ids)
            valid_tids = list(tissue_labels.keys())
            valid_tids = [tid for tid in valid_tids if tid in tile_tokens]
            
            # get train and test patient IDs
            train_pids = [pid for pid in patient_ids if patient_split_dict[pid] == "train"]
            test_pids = [pid for pid in patient_ids if patient_split_dict[pid] == "test"]

            logger.info(f"Evaluating test fold of spora-io for target '{label_col}'...")
            train_val_tids = [tid for tid in valid_tids if tid.split("_")[1] in train_pids]
            test_tids = [tid for tid in valid_tids if tid.split("_")[1] in test_pids]

            # encode labels for train tids to do stratified split for validation set
            y_train_full = [tissue_labels_encoded[tid] for tid in train_val_tids]
            val_splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=config.training.validation_fraction,
                random_state=config.random_seed,
            )

            train_sub_idx, val_sub_idx = next(val_splitter.split(train_val_tids, y_train_full))
            train_tids = [train_val_tids[i] for i in train_sub_idx]
            val_tids = [train_val_tids[i] for i in val_sub_idx]

            logger.info(
                f"Target '{label_col}' - Train/Val/Test samples: {len(train_tids)}/{len(val_tids)}/{len(test_tids)}"
            )
            
            model, test_preds, test_labels = abmil_train_test(
                config,
                tile_tokens,
                tissue_labels_encoded,
                train_tids,
                val_tids,
                test_tids,
            )
            all_gts = list(test_labels)
            all_preds = list(test_preds)

            # Calculate and save evaluation metrics
            logger.info(f"Evaluating results for target '{label_col}'...")
            logger.info(f"F1 Score: {f1_score(all_gts, all_preds, average='weighted'):.4f} - Acc: {(np.array(all_gts) == np.array(all_preds)).mean():.4f}")

            report = classification_report(
                    y_true=all_gts, 
                    y_pred=all_preds,
                    labels=label_encoder.transform(label_encoder.classes_),
                    target_names=label_encoder.classes_,
                    output_dict=True,
                )
            report = transform_classification_report_to_df(report)
            report = report.assign(model=config.model.model_name, task=label_col)
            report.to_parquet(results_dir / f'{dataset_name}_{label_col}_classification_report.parquet')

            bootstrap_report = bootstrap_classification_report(
                y_true=all_gts, 
                y_pred=all_preds, 
                n_bootstraps=1000, 
                labels=label_encoder.transform(label_encoder.classes_),
                target_names=label_encoder.classes_,
                random_state=42
            )
            bootstrap_report = transform_bootstrap_report_to_df(bootstrap_report, n_bootstraps=1000)
            bootstrap_report = bootstrap_report.assign(model=config.model.model_name, task=label_col)
            bootstrap_report.to_parquet(results_dir / f'{dataset_name}_{label_col}_bootstrap_classification_report.parquet')

            cm = confusion_matrix(y_true=all_gts, y_pred=all_preds, labels=label_encoder.transform(label_encoder.classes_))
            cm = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
            cm.to_parquet(results_dir / f'{dataset_name}_{label_col}_confusion_matrix.parquet')


if __name__ == "__main__":

    cli_config = OmegaConf.from_cli()
    config = OmegaConf.load('configs/base_config.yaml')
    downstream_model = OmegaConf.load('configs/downstream_models/abmil.yaml')

    model_config = OmegaConf.load(cli_config.model_config)

    datasets_config = load_multiple_configs(cli_config.datasets_config)
    benchmarks_config = load_multiple_configs(cli_config.benchmarks_config)

    config = OmegaConf.merge(config, downstream_model, model_config, datasets_config, benchmarks_config, cli_config)

    logger.info(f'Config: \n{OmegaConf.to_yaml(config)}')

    set_seed(config.random_seed)
    run_spora_split_validation(config)