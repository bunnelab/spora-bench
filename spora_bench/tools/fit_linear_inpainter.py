import os
from pathlib import Path

import numpy as np
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

import joblib
try:
    from cuml.linear_model import LinearRegression
except Exception as e:
    logger.warning("Could not import cuML. Make sure you have cuML installed and a compatible GPU. Falling back to CPU implementation.")
    from sklearn.linear_model import LinearRegression

from spora_bench.utils.setup_utils import load_multiple_configs, set_seed
from spora_io.datasets import MultiplexImagingDataset

def run_training(config):

    output_dir = Path(config.output_dir)
    model_dir = output_dir / config.model.model_name

    for dataset_key, dataset_config in config.datasets.items():
        dataset_name = dataset_config.name

        logger.info(f"Training on dataset: {dataset_key} ({dataset_name})")

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

        subsample_fraction = float(config.get('subsample_fraction', 0.1))

        train_tissue_ids = dataset.tissue_modality_metadata[
            dataset.tissue_modality_metadata['split'] == 'train'
        ].index.values

        logger.info(f"Found {len(train_tissue_ids)} training tissues.")

        # Build feature matrix (N,C) assuming canoncical and fixed order of channels across tissues.
        reference_channel_names = None # canonical channel ordering for the dataset, established from the first tissue
        collected_pixels = []

        for tissue_id in tqdm(train_tissue_ids, desc=f"Collecting pixels for dataset {dataset_key}"):
            tissue = dataset.get_tissue(tissue_id, kind="uniprot_filtered", preprocess=True, image_mode="CHW")

            image = np.asarray(tissue.image)                 # (C, H, W)
            tissue_channel_names = np.asarray(tissue.channel_names)

            # Establish the canonical channel ordering from the first tissue, and compare every subsequent tissue to it.
            if reference_channel_names is None:
                reference_channel_names = tissue_channel_names
                logger.info(f"Established canonical channel ordering for dataset {dataset_key}: {reference_channel_names}")
                if len(set(map(str, reference_channel_names))) != len(reference_channel_names):
                    logger.warning(
                        "Duplicate channel names detected in the channel set. Models are keyed by "
                        "channel name, so duplicates will collide and overwrite each other."
                    )

            elif not np.array_equal(tissue_channel_names, reference_channel_names):
                logger.warning(
                    f"Tissue {tissue_id} has a different channel set than the reference; skipping it."
                )
                continue

            C = image.shape[0]
            flat = image.reshape(C, -1).T                    # (H*W, C)

            # subsample ~`subsample_fraction` of this tissue's pixels
            if 0.0 < subsample_fraction < 1.0:
                n_pixels = flat.shape[0]
                n_sub = max(1, int(round(n_pixels * subsample_fraction)))
                idx = np.random.choice(n_pixels, size=n_sub, replace=False)
                flat = flat[idx]

            collected_pixels.append(flat)

        if not collected_pixels:
            raise ValueError(f"No usable training tissues found for dataset '{dataset_name}'.")

        all_pixels = np.concatenate(collected_pixels, axis=0)  # (N, C)
        logger.info(
            f"Collected {all_pixels.shape[0]} pixels x {all_pixels.shape[1]} channels "
            f"(subsample_fraction={subsample_fraction})."
        )

        # Fit one linear regression model per marker, using all other markers as features.
        regression_models = {}
        n_channels = len(reference_channel_names)
        for cidx in tqdm(range(n_channels), desc="Training per-marker regressors"):
            target_channel = str(reference_channel_names[cidx])

            feature_mask = np.ones(n_channels, dtype=bool)
            feature_mask[cidx] = False

            X = all_pixels[:, feature_mask]   # (N, C-1) -- all other markers, canonical order
            y = all_pixels[:, cidx]           # (N,)     -- the target marker

            model = LinearRegression()
            model.fit(X, y)
            regression_models[target_channel] = model
        logger.debug(f"Fitted regressor for {target_channel} on {X.shape}.")

        # Save the trained models to disk.
        if config.model.checkpoint_path is None:
            checkpoint_dir = model_dir / 'checkpoints' / dataset_key
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = checkpoint_dir / "regression_models.joblib"
        else:
            assert len(config.datasets) == 1, "When using a custom checkpoint path, only one dataset should be specified."
            checkpoint_path = Path(config.model.checkpoint_path)
            os.makedirs(checkpoint_path.parent, exist_ok=True)

        joblib.dump(regression_models, checkpoint_path)
        logger.info(
            f"Finished training '{config.model.model_name}' on dataset '{dataset_name}'. "
            f"Saved {len(regression_models)} linear regression models to {checkpoint_path}."
        )


if __name__ == "__main__":
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.load('configs/base_config.yaml')

    model_config = OmegaConf.load(cli_config.model_config)

    datasets_config = load_multiple_configs(cli_config.datasets_config)
    config = OmegaConf.merge(config, model_config, datasets_config, cli_config)

    logger.info(f'Config: \n{OmegaConf.to_yaml(config)}')

    set_seed(config.random_seed)
    run_training(config)