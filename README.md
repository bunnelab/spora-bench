<img src=".github/banner-spora-bench.png" alt="Spora Bench" width="45%" style="float:left;" />
<div style="clear:both;"></div>

# Introduction
**spora [bench]** is a benchmark for spatial proteomics foundation models. Building upon the unified dataset of **[spora [data]](TBD)** and the data interface **[spora [io]](https://github.com/bunnelab/spora-io)**, it provides standardized evaluation tasks and protocols to assess and compare spatial proteomics foundation models across scales: 

At the *subcellular-level* virtual staining assess a model's learnt understanding of complex marker co-localization patterns. At the *cell-level*, coarse- and fine-grained cell phenotyping tasks evaluate the local biological information content captured by representations via standardized logistic regression. At the *tissue-level*, pathology tasks (cancer grading, subtyping and treatment response prediction), test a model's abilities to capture clinically relevant local and global tissue features. 

For more information about **spora**, please also refer to the following ressources:
1. Our paper *To be announced*
2. Our project website: *To be announced*
3. Documentation: *To be announced*

# ⚙️ Installation
Please first install our data loading interface, **[spora-io](https://github.com/bunnelab/spora-io)**:
```bash
git clone https://github.com/bunnelab/spora-io.git
cd spora-io
pip install -e .
```
Next, clone the repository of **spora [bench]**:
```bash 
git clone https://github.com/bunnelab/spora-bench.git
cd spora-bench
pip install -r requirements.txt
```
Finally, ensure that the foundation models of interest are properly set up. For installation instructions, please refer to their respective repositories:
1. VirTues: [https://github.com/bunnelab/virtues](https://github.com/bunnelab/virtues) 
2. KRONOS: [https://github.com/mahmoodlab/KRONOS](https://github.com/mahmoodlab/KRONOS)

spora [bench] can also be easily extended to evaluate new foundation models. For instructions on how to integrate a new model, please refer to the corresponding section below.

# 📈 Running benchmarks

## Downloading datasets

With **spora [data]**, we provide a comprehensive collection of standardized datasets, including benchmark tasks. You can find all available datasets in our [Dataset Browser](TBD).

Datasets can be downloaded using `rclone`. For detailed instructions, please refer to our documentation.

## Overview of benchmark pipelines

**spora [bench]** consists of four benchmark pipelines:

1. `run_cell_level_tasks` for cell phenotyping tasks,
2. `run_tissue_level_tasks` for tissue-level pathology tasks,
3. `run_virtual_staining_tasks` for virtual staining tasks (multiplex-to-multiplex inpainting), and
4. `run_he_virtual_staining_tasks` for H&E-to-multiplex virtual staining tasks.

These pipelines are configured and executed using a modular system of `.yaml` configuration files. We distinguish between four types of configuration files:

1. The `base_config`, located at `configs/base_config`, specifies global parameters such as the output directory and random seed for reproducibility.
2. The `model_config` stores all information required to instantiate the target model.
3. The `datasets_config` stores all information and parameters required to instantiate the target datasets, e.g., storage locations and standardization settings.
4. The `benchmarks_config` stores all information about which tasks to run for each dataset.

## Quick Configuration

For the purpose of simply running the benchmarks listed in the paper, all required `.yaml` configurations are provided in `configs/`, and only a few path variables need to be updated:

1. Update the `output_dir` field in `configs/base_config`. Results will be written to subdirectories of `output_dir/<model_name>/`.
2. In all dataset configs (e.g. `configs/datasets/virtues/cords2023cancer.yaml`), update the `path` field to point to the root directory of the respective dataset.
3. In the model config (e.g. `configs/models/virtues.yaml`), update all required model-specific path variables (e.g. `checkpoint_path` and `marker_embeddings_dir` for VirTues; `checkpoint_path` and `marker_metadata_path` for KRONOS).

## Cell phenotyping

### Starting an individual benchmark

To execute cell phenotyping benchmarks, run:
```
python -m spora_bench.benchmarks.run_cell_level_tasks model_config=<path-to-model-config> datasets_config=<path-to-datasets-config> benchmarks_config=<path-to-benchmarks-config>
```
Results (confusion matrix, classification report, and bootstrapped classification report) will be saved to `output_dir/<model-name>/results`.

To give an example, the following command runs the cell phenotyping benchmark for VirTues on the `hoch2022multiplexed` dataset:
```
python -m spora_bench.benchmarks.run_cell_level_tasks model_config=configs/models/virtues.yaml datasets_config=configs/datasets/virtues/hoch2022multiplexed.yaml benchmarks_config=configs/benchmarks/cell_level/hoch2022multiplexed.yaml
```
**Note:** Some models, for instance KRONOS, might require distinct dataset configs to account for differences in standardization procedures. For KRONOS you find the default dataset configs in `configs/datasets/kronos/`.

### Batched submission of benchmark tasks
To run multiple benchmarks at once, you may pass to `datasets_config` and `benchmarks_config` not only a single path but also paths containing wildcards and lists of paths. For instance, to run all cell phenotyping tasks for VirTues, execute
```
python -m spora_bench.benchmarks.run_cell_level_tasks model_config=configs/models/virtues.yaml datasets_config=configs/datasets/virtues/*.yaml benchmarks_config=configs/benchmarks/cell_level
```

## Tissue-level pathology tasks
Similarly to the cell phenotyping tasks, tissue-level benchmarks are executed via the command:
```
python -m spora_bench.benchmarks.run_tissue_level_tasks model_config=<path-to-model-config> datasets_config=<path-to-datasets-config> benchmarks_config=<path-to-benchmarks-config>
```
For example, the following command runs all tissue-level tasks on `cords2024cancer` using VirTues:
```
python -m spora_bench.benchmarks.run_tissue_level_tasks model_config=configs/models/virtues.yaml datasets_config=configs/datasets/virtues/cords2024cancer.yaml benchmarks_config=configs/benchmarks/tissue_level/cords2024cancer.yaml
```
Similar to the cell phenotyping pipeline, batched submission of datasets and benchmark configs is possible via lists or wildcards.

## Virtual staining tasks
Virtual staining tasks can be executed via the following command:
```
python -m spora_bench.benchmarks.run_virtual_staining_tasks model_config=<path-to-model-config> datasets_config=<path-to-datasets-config>
```
No additional benchmark config is required here. For example the following command runs virtual staining on `rigamonti2024integrating`:
```
python -m spora_bench.benchmarks.run_virtual_staining_tasks model_config=configs/models/virtues.yaml datasets_config=configs/datasets/virtues/rigamonti2024integrating.yaml
```
Again, the pipeline accepts batched submission via lists or wildcards passed to `datasets_config`.

## H&E virtual staining tasks
This pipeline evaluates models that translate H&E directly into multiplex marker maps, with
no multiplex input at all (e.g. ROSIE, HistoPlexer, GigaTIME) — unlike the virtual staining
pipeline above, which inpaints a dropped multiplex channel from the *other* multiplex channels.
Each model is scored on whatever markers it shares with the dataset (dataset ∩ model),
independently of what any other model being run alongside it does or doesn't support.
`model_config` accepts a single path, a wildcard, or a list of model config paths; when multiple
models are listed they are all instantiated in the same run purely so their predictions over the
dataset's tiles can be computed together (e.g. so ROSIE's cross-tile batching still applies) —
it has no effect on which markers get scored for any individual model. It can be executed via:
```
python -m spora_bench.benchmarks.run_he_virtual_staining_tasks model_config=<path-or-list-to-model-configs> datasets_config=<path-to-datasets-config>
```
For example, the following command runs ROSIE, HistoPlexer and GigaTIME on `lin2023high`:
```
python -m spora_bench.benchmarks.run_he_virtual_staining_tasks model_config=[configs/models/rosie.yaml,configs/models/histoplexer.yaml,configs/models/gigatime.yaml] datasets_config=configs/datasets/he_staining/lin2023high.yaml
```
No additional benchmark config is required. `n_crops` (number of randomly sampled tiles per
dataset; default 1000) may be overridden via CLI, e.g. `n_crops=200`. Results (per-crop
Pearson/MSE/SSIM and their mean/std summary across crops) are saved per model to
`output_dir/<model-name>/results/`.

Finally, baseline auto-correlation values between markers can be computed:
```
python -m spora_bench.tools.compute_correlations dataset_config=<path-to-datasets-config>
```
# 📋 Task Zoo

In the following you find a lists benchmark tasks, for which results are reported in our paper.

## Cell phenotyping tasks
| Dataset |  Dataset config (VirTues) | Benchmark config |
| --- | --- | --- | 
| cords2023cancer | `configs/datasets/virtues/cords2023cancer.yaml` | `configs/benchmarks/cell_level/cords2023cancer.yaml` |
| cords2024cancer | `configs/datasets/virtues/cords2024cancer.yaml` | `configs/benchmarks/cell_level/cords2024cancer.yaml` |
| danenberg2022breast | `configs/datasets/virtues/danenberg2022breast.yaml` | `configs/benchmarks/cell_level/danenberg2022breast.yaml` |
| hoch2022multiplexed | `configs/datasets/virtues/hoch2022multiplexed.yaml` | `configs/benchmarks/cell_level/hoch2022multiplexed.yaml` |
| lin2023highsubset | `configs/datasets/virtues/lin2023highsubset.yaml` | `configs/benchmarks/cell_level/lin2023highsubset.yaml` | 
| meyer2025stratification | `configs/datasets/virtues/meyer2025stratification.yaml` | `configs/benchmarks/cell_level/meyer2025stratification.yaml` |
| moldoveanu2022spatially | `configs/datasets/virtues/moldoveanu2022spatially.yaml` | `configs/benchmarks/cell_level/moldoveanu2022spatially.yaml` |
| rigamonti2024integrating | `configs/datasets/virtues/rigamonti2024integrating.yaml` | `configs/benchmarks/cell_level/rigamonti2024integrating.yaml` |
| schulz2024immucan |  `configs/datasets/virtues/schulz2024immucan.yaml` | `configs/benchmarks/cell_level/schulz2024immucan.yaml` | 
| wang2023spatial | `configs/datasets/virtues/wang2023spatial.yaml` | `configs/benchmarks/cell_level/wang2023spatial.yaml` | 

## Tissue level tasks
| Dataset |  Dataset config (VirTues) | Benchmark config |
| --- | --- | --- | 
| cords2024cancer | `configs/datasets/virtues/cords2024cancer.yaml` | `configs/benchmarks/cell_level/cords2024cancer.yaml` |
| danenberg2022breast | `configs/datasets/virtues/danenberg2022breast.yaml` | `configs/benchmarks/cell_level/danenberg2022breast.yaml` |
| fischer2023multiplex | `configs/datasets/virtues/fischer2023multiplex.yaml` | `configs/benchmarks/cell_level/fischer2023multiplex.yaml` |
| hoch2022multiplexed | `configs/datasets/virtues/hoch2022multiplexed.yaml` | `configs/benchmarks/cell_level/hoch2022multiplexed.yaml` |
| meyer2025stratification | `configs/datasets/virtues/meyer2025stratification.yaml` | `configs/benchmarks/cell_level/meyer2025stratification.yaml` |
| wang2023spatial | `configs/datasets/virtues/wang2023spatial.yaml` | `configs/benchmarks/cell_level/cords2023cancer.yaml` |

## Virtual staining tasks
| Dataset | Dataset config (VirTues) | 
| --- | --- | 
| cords2024cancer | `configs/datasets/virtues/cords2024cancer.yaml` |
| schulz2024immucan | `configs/datasets/virtues/schulz2024immucan.yaml` |
| rigamonti2024integrating | `configs/datasets/virtues/rigamonti2024integrating.yaml` |
| danenberg2022breast | `configs/datasets/virtues/danenberg2022breast.yaml` | 

## H&E virtual staining tasks
| Dataset | Dataset config |
| --- | --- |
| lin2023high | `configs/datasets/he_staining/lin2023high.yaml` |
| hickey2023organization | `configs/datasets/he_staining/hickey2023organization.yaml` |
| schurch2020coordinated | `configs/datasets/he_staining/schurch2020coordinated.yaml` |
| phillips2021immune | `configs/datasets/he_staining/phillips2021immune.yaml` |

# 🤖 Model Zoo
Our repository contains implementations of two spatial proteomics foundation models, a ResNet baseline, astir, and three H&E-to-multiplex virtual staining baselines:
| Model | Model config | Supported Tasks |
| --- | --- | --- |
| VirTues | `configs/models/virtues.yaml` | cell phenotyping, tissue-level task and virtual staining |
| KRONOS | `configs/models/kronos.yaml` | cell phenotyping and tissue-level task |
| ResNet | `configs/models/resnet.yaml` | only tissue-level task |
| astir | `configs/models/astir.yaml` | only cell phenotyping |
| ROSIE | `configs/models/rosie.yaml` | only H&E virtual staining |
| HistoPlexer | `configs/models/histoplexer.yaml` | only H&E virtual staining |
| GigaTIME | `configs/models/gigatime.yaml` | only H&E virtual staining |

**Note:** To run astir, we provide a separate pipeline `spora_bench/benchmarks/run_astir.py` for which benchmark configs are located in `configs/benchmarks/astir`. Similarly, ROSIE/HistoPlexer/GigaTIME run via the separate `spora_bench/benchmarks/run_he_virtual_staining_tasks.py` pipeline described above.

# 🛠️ Setting up new datasets, tasks and models
spora [bench] is designed to be modular benchmark system that can be easily extended by new datasets, tasks and models.

## Setting up a new dataset
spora [bench] is built upon spora [data] and spora [io]. Any dataset contained in spora [data], can be easily integrating into spora [bench] by setting up a corresponding `.yaml` dataset  config. This dataset config should adhere to the following format:
```
datasets:
  <dataset_name>:
    name: <dataset_name>
    path: <path/to/dataset>
    modality: <modality> 
    standardization: quantile_clipping_log1p/uq_0.99_image # path to standardization metadata relative to modality directory, e.g. quantile_clipping_log1p/uq_0.99_image for 99%-clipping and lop1p transformation
    resolution: <resolution> # resolution in mikrons/ppx default: 1.0
    use_mean_std: <True/False> # value depends on the standardization
    disable_quantile_mask: <True/False> # option to filter out channels with zero quantiles and/or zero standard deviations
```
For examples, please refer to `configs/datasets/virtues`.

The benchmark pipeline can also be applied to novel or proprietary datasets not contained in spora [data], provided they are processed into the standardized format defined by spora [data]. For detailed formatting instructions, please refer to the spora [data] documentation.

## Setting up a new benchmark
To setup a new cell phenotyping or tissue-level benchmark task, simply create a `.yaml` benchmark config in one of the following two formats. Virtual staining task do not require additional setup of configs. 

### Cell-phenotyping tasks
```
datasets:
  <dataset_name>:
    benchmarks:
      cell_level: 
        <task_name>:
          label_col: <column_name>
          excluded_classes: <list of classes to exclude>
```
### Tissue-level pathology tasks
```
datasets:
  <dataset_name>:
    benchmarks:
      tissue_level:
        <task_name>:
          label_col: <column_name>
          excluded_classes: <list of classes to exclude>
```
The field `label_col` should correspond to the column in `metadata/cells.parquet` or `metadata/tissues.parquet` that contains the target labels, respectively. Classes listed in `excluded_classes` are filtered out prior to training and evaluation.

## Setting up a new model

### Model implementation

We provide a model wrapping class `spora_bench.wrapper.SporaModelWrapper` that defines the signatures of the following methods expected by our pipelines:

| Method | Benchmarks Requiring Implementation | 
| --- | --- | 
| `compute_cell_tokens` | cell phenotyping | 
| `embed_tissue` | tissue-level | 
| `postprocess_tile_embeddings` | tissue-level (optional) | 
| `predict_marker` | virtual staining |
| `predict_markers_from_he` | H&E virtual staining |
| `predict_markers_from_he_batch` | H&E virtual staining (optional; override for cross-tile batched inference) |

For details on the expected input arguments and return types of these methods, please refer to our documentation or the `SporaModelWrapper` class itself.

To implement a new model, create a class that inherits from `SporaModelWrapper`. The custom model class must:
* implement all interface methods required for the benchmarks you intend to run,
* define an `__init__` constructor that:
  * fully initializes the wrapped model,
  * calls `super().__init__(model_name=<model-name>)`, and
  * accepts keyword arguments only; positional arguments are not supported.

Please refer to `models/virtues/virtues_model.py`, `models/kronos_model.py` and `models/resnet/resnet.py` for examples.

### Config setup
Once you have implemented your model wrapping class, it only remains to setup the corresponding `.yaml` model config. This model config is used to instantiate your model via `hydra.utils.instantiate` and must therefore follow the format:
```
model:
  _target_: path.to.model.MyModelClass
  keywordarg1: <first-keyword-argument>
  keywordarg2: <second-keyword-argument>
  keywordarg3: <third-keyword-argument>
  ...
  keywordargn: <n-th-keyword-argument>
```

# 📝 Citation
*To be announced*
