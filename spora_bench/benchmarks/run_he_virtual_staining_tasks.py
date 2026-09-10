"""H&E -> multiplex marker virtual staining benchmark.

This pipeline evaluates models that translate H&E directly into multiplex marker maps, with no multiplex input at all
(e.g. ROSIE, HistoPlexer, GigaTIME). Evaluation is per-tile (via `spora_io.SporaDataset`,
which can jointly sample the `he` and a multiplex modality from the same tile) rather than
per-tissue, and markers are canonicalized (see `spora_bench.utils.virtual_staining_utils`) so
that models with different channel vocabularies can be matched against a dataset's own
vocabulary regardless of naming differences.

Each model is scored on whatever markers it shares with the dataset, fully
independently of the other models being run alongside it, there is no additional restriction
to markers shared across all models. `model_config` accepts a single path, wildcard, or list of
model config paths and instantiates all of them in one run purely so their per-tile predictions
can be computed together (e.g. so ROSIE's cross-tile batching still applies); it does not affect
which markers get scored for any individual model.
"""
import os
from glob import glob
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from spora_bench.utils.setup_utils import load_multiple_configs, set_seed
from spora_bench.utils.virtual_staining_utils import build_marker_index, canon, crop_metrics, resize2d
from spora_io import SporaDataset

TILE = 256


def load_model_configs(model_config_paths: Union[str, List[str]]) -> List[DictConfig]:
    """Loads one or more model config files as SEPARATE configs since every model config shares the same top-level `model` key
    and merging would let the last one silently overwrite the rest. Supports the same
    single-path / list-of-paths / glob-wildcard flexibility as `load_multiple_configs`.
    """
    if isinstance(model_config_paths, str):
        model_config_paths = [model_config_paths]

    configs = []
    for path in model_config_paths:
        extended_paths = sorted(glob(path, recursive=True))
        for extended_path in extended_paths:
            configs.append(OmegaConf.load(extended_path))

    if not configs:
        raise ValueError(f"No model configuration files found for the provided paths: {model_config_paths}")
    return configs


def build_dataset(dataset_config: DictConfig) -> SporaDataset:
    modality_kwargs = {
        dataset_config.modality: {
            "standardization": dataset_config.standardization,
            "replace_nuclear_uniprot_ids": dataset_config.get("replace_nuclear_uniprot_ids", True),
            "use_mean_std": dataset_config.use_mean_std,
            "disable_quantile_mask": dataset_config.disable_quantile_mask,
        },
        "he": {"mean_std_type": "imagenet"},
    }
    dataset_root = Path(dataset_config.path)
    return SporaDataset(
        dataset_names=[dataset_config.name],
        datasets_dir=dataset_root.parent,
        modalities=["he", dataset_config.modality],
        resolution=dataset_config.resolution,
        tile_size=TILE,
        modality_kwargs=modality_kwargs,
    )


def sample_crops(ds: SporaDataset, mp_key: str, n: int, max_tries_factor: int = 50) -> List[Dict]:
    """Samples n dicts {idx, stem, he_tensor, gt(C,H,W), names, mask} via random tiles.

    sample_random_tile() can return tiles missing a modality (or with a None image);
    those are skipped and re-drawn. Gives up after n*max_tries_factor attempts."""
    got, tries, max_tries = 0, 0, n * max_tries_factor
    crops = []
    while got < n and tries < max_tries:
        tries += 1
        s = ds.sample_random_tile()
        mods = s.get("modalities", {})
        he_t = mods.get("he")
        mx = mods.get(mp_key)
        if he_t is None or mx is None:
            continue
        he = getattr(he_t, "image", None)
        gt_t = getattr(mx, "image", None)
        if he is None or gt_t is None:
            continue
        names = list(getattr(mx, "channel_names", []))
        if not names:
            continue
        gt = gt_t.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(gt_t) \
            else np.asarray(gt_t, dtype=np.float32)
        mask = getattr(mx, "image_loading_mask", None)
        if mask is None:
            mask = getattr(mx, "measured_mask", np.ones(len(names), bool))
        mask = np.asarray(mask, bool)
        he_tensor = he.float() if torch.is_tensor(he) else torch.from_numpy(np.asarray(he, dtype=np.float32))

        stem = f"{s['tissue_id']}_tile{s['tile_id']}_{got:05d}"
        crops.append({"idx": got, "stem": stem, "he_tensor": he_tensor, "gt": gt, "names": names, "mask": mask})
        got += 1

    if got < n:
        logger.warning(f"only {got}/{n} valid crops after {tries} draws (missing he/{mp_key} modality).")
    return crops


def evaluate_dataset(dataset_key: str,
                     dataset_config: DictConfig,
                     models: List,
                     n_crops: int,
                     ) -> Dict[str, pd.DataFrame]:
    """Returns {model_name: per-crop long-form results DataFrame}.

    Each model is scored on its own dataset ∩ model marker intersection, independently of
    what the other models in `models` do or don't support."""
    mp_key = dataset_config.modality
    ds = build_dataset(dataset_config)
    crops = sample_crops(ds, mp_key, n_crops)
    empty = {m.model_name: pd.DataFrame() for m in models}
    if not crops:
        return empty

    ds_markers = set()
    for c in crops:
        ds_markers |= set(build_marker_index(c["names"]))

    model_scored = {
        m.model_name: sorted(ds_markers & set(getattr(m, "supported_he_markers", set())))
        for m in models
    }
    for m in models:
        logger.info(f"[{dataset_key}] {m.model_name} scored markers "
                    f"({len(model_scored[m.model_name])}): {model_scored[m.model_name]}")
    if not any(model_scored.values()):
        return empty

    he_tiles = [c["he_tensor"] for c in crops]
    
    preds_by_model = {}
    for m in models:
        shared = model_scored[m.model_name]
        if not shared:
            preds_by_model[m.model_name] = [{} for _ in crops]
            continue
        logger.info(f"[{dataset_key}] predicting with {m.model_name} over {len(crops)} crops, "
                    f"{len(shared)} target markers ...")
        preds_by_model[m.model_name] = m.predict_markers_from_he_batch(he_tiles, target_markers=shared)

    rows = []
    for ci, c in enumerate(tqdm(crops, desc=f"[{dataset_key}] metrics")):
        gt, names, mask = c["gt"], c["names"], c["mask"]
        H, W = gt.shape[-2:]

        valid = {canon(names[j]): j for j in range(len(names))
                if mask[j] and canon(names[j]) is not None}

        for m in models:
            for mk in model_scored[m.model_name]:
                if mk not in valid:
                    continue
                src = preds_by_model[m.model_name][ci].get(mk)
                if src is None:
                    continue
                gt2d = gt[valid[mk]]
                src_np = src.detach().cpu().numpy() if torch.is_tensor(src) else np.asarray(src)
                pred2d = resize2d(src_np, (H, W))
                pear, mse, ss = crop_metrics(pred2d, gt2d)
                rows.append({
                    "cohort": dataset_key, "model": m.model_name, "marker": mk,
                    "crop": c["idx"], "pearson": pear, "mse_norm": mse, "ssim": ss,
                })

    long_df = pd.DataFrame(rows)
    return {
        m.model_name: long_df[long_df["model"] == m.model_name].reset_index(drop=True)
        for m in models
    }


def run_he_virtual_stainings(config: DictConfig, model_configs: List[DictConfig]):
    output_dir = Path(config.output_dir)
    n_crops = config.get("n_crops", 1000)

    models = [instantiate(mc.model) for mc in model_configs]

    for dataset_key, dataset_config in config.datasets.items():
        logger.info(f"Processing dataset: {dataset_key}")
        per_model_long = evaluate_dataset(dataset_key, dataset_config, models, n_crops)

        for m in models:
            long_df = per_model_long[m.model_name]
            if long_df.empty:
                logger.info(f"No scored markers for dataset {dataset_key}, model {m.model_name}. Skipping.")
                continue

            results_dir = output_dir / m.model_name / "results"
            os.makedirs(results_dir, exist_ok=True)

            long_df.to_parquet(results_dir / f"{dataset_key}_he_virtual_staining_percrop.parquet")

            summary = (long_df.groupby(["cohort", "model", "marker"])
                      .agg(pearson_mean=("pearson", "mean"), pearson_std=("pearson", "std"),
                           mse_mean=("mse_norm", "mean"), mse_std=("mse_norm", "std"),
                           ssim_mean=("ssim", "mean"), ssim_std=("ssim", "std"),
                           n=("crop", "count"))
                      .reset_index())
            summary.to_parquet(results_dir / f"{dataset_key}_he_virtual_staining_summary.parquet")

            logger.info(f"Finished H&E virtual staining benchmark for dataset {dataset_key}, "
                        f"model {m.model_name}. Results saved to {results_dir}")


if __name__ == "__main__":
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.load('configs/base_config.yaml')

    model_configs = load_model_configs(cli_config.model_config)
    datasets_config = load_multiple_configs(cli_config.datasets_config)
    config = OmegaConf.merge(config, datasets_config, cli_config)

    logger.info(f'Config: \n{OmegaConf.to_yaml(config)}')

    set_seed(config.random_seed)
    run_he_virtual_stainings(config, model_configs)
