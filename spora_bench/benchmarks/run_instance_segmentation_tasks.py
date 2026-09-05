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

from hydra.utils import instantiate

THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def run_instance_segmentations(
        config: DictConfig
    ):
    """
    Evaluates the model on instance segmentation benchmarks.
    Args:
        config: The configuration object containing model, and dataset information.
    """
    output_dir = Path(config.output_dir)
    results_dir = output_dir / config.model.model_name / 'results'
    os.makedirs(results_dir, exist_ok=True)

    spora_model = instantiate(config.model)

    for dataset_key, dataset_config in config.datasets.items():
        logger.info(f'Running instance segmentation benchmark for dataset: {dataset_key}')
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

        all_stats = []
        test_tissue_ids = dataset.tissue_modality_metadata[dataset.tissue_modality_metadata['split'] == 'test'].index.values

        for tissue_id in tqdm(test_tissue_ids, desc=f"Processing tissues for dataset {dataset_key}"):

            tissue = dataset.get_tissue(tissue_id, kind="uniprot_filtered", preprocess=True, image_mode="CHW")
            true_cell_mask = dataset.get_cell_instance_mask(tissue_id,).mask.numpy()

            # TODO check this
            if tissue.image.shape[1] < 256 or tissue.image.shape[2] < 256:
                logger.warning(f"Tissue {tissue_id} has image size {tissue.image.shape[1:]} which is smaller than 256x256. Skipping this tissue.")
                continue

            predicted_mask = spora_model.predict_instance_segmentation(tissue).cpu().numpy()

            assert true_cell_mask.shape == predicted_mask.shape, (
                f"Shape mismatch for {tissue_id}: {true_cell_mask.shape} vs {predicted_mask.shape}"
            )

            match_stats = compute_matches(true_cell_mask, predicted_mask, iou_thresholds=THRESHOLDS)
            all_stats.extend(match_stats)

        if not all_stats:
            logger.error(f"No valid tissues processed for dataset {dataset_key}. Skipping results aggregation and saving.")
            continue

        total_tp_per_threshold = defaultdict(int)
        total_fp_per_threshold = defaultdict(int)
        total_fn_per_threshold = defaultdict(int)
        for thr, tp, fp, fn in all_stats:
            total_tp_per_threshold[thr] += tp
            total_fp_per_threshold[thr] += fp
            total_fn_per_threshold[thr] += fn

        results = []
        for thr in total_tp_per_threshold.keys():
            tp = total_tp_per_threshold[thr]
            fp = total_fp_per_threshold[thr]
            fn = total_fn_per_threshold[thr]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            results.append({
                "dataset": dataset_key,
                "model": config.model.model_name,
                "metric": "f1-score",
                'score': f1_score,
                "iou_threshold": thr,
            })
            results.append({
                "dataset": dataset_key,
                "model": config.model.model_name,
                "metric": "precision",
                'score': precision,
                "iou_threshold": thr,
            })
            results.append({
                "dataset": dataset_key,
                "model": config.model.model_name,
                "metric": "recall",
                'score': recall,
                "iou_threshold": thr,
            })

        results_df = pd.DataFrame(results)
        results_df.to_parquet(results_dir / f"{dataset_key}_instance_segmentation_results.parquet")
        logger.info(f'Finished instance segmentation benchmarks for dataset: {dataset_key}. Results saved to {results_dir / f"{dataset_key}_instance_segmentation_results.parquet"}')
        

if __name__ == "__main__":
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.load('configs/base_config.yaml')

    model_config = OmegaConf.load(cli_config.model_config)

    datasets_config = load_multiple_configs(cli_config.datasets_config)

    # Benchmark configs seems to be not needed for now.
    # benchmarks_config = load_multiple_configs(cli_config.benchmarks_config)
    # config = OmegaConf.merge(config, model_config, datasets_config, benchmarks_config, cli_config)
    config = OmegaConf.merge(config, model_config, datasets_config, cli_config)

    logger.info(f'Config: \n{OmegaConf.to_yaml(config)}')

    set_seed(config.random_seed)
    run_instance_segmentations(config)