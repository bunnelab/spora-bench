Running Benchmarks
==================

``spora-bench`` provides modular command-line entrypoints for the benchmark
pipelines described in the README. The scripts live under
``spora_bench.benchmarks`` and are driven by YAML config files.

Required config types
---------------------

- ``base_config``: global settings such as ``output_dir`` and the random seed.
- ``model_config``: parameters needed to construct the target model.
- ``datasets_config``: dataset paths and preprocessing details.
- ``benchmarks_config``: which benchmark tasks to execute.

Quick setup
-----------

For the benchmark runs reported in the paper, update the path fields in the
configs under ``configs/``:

1. Set ``output_dir`` in ``configs/base_config``.
2. Set the dataset root paths in the dataset configs, for example
   ``configs/datasets/cords2023cancer.yaml``.
3. Set the model-specific paths in the model configs, for example
   ``configs/models/virtues.yaml``.

Cell phenotyping
----------------

Run a single cell phenotyping benchmark with:

.. code-block:: bash

   python -m spora_bench.benchmarks.run_cell_level_tasks \
     model_config=<path-to-model-config> \
     datasets_config=<path-to-datasets-config> \
     benchmarks_config=<path-to-benchmarks-config>

For example, to run VirTues on ``hoch2022multiplexed``:

.. code-block:: bash

   python -m spora_bench.benchmarks.run_cell_level_tasks \
     model_config=configs/models/virtues.yaml \
     datasets_config=configs/datasets/virtues/hoch2022multiplexed.yaml \
     benchmarks_config=configs/benchmarks/cell_level/hoch2022multiplexed.yaml

You can also pass wildcards or lists of config paths to run several cell-level
tasks in one command.

Tissue-level pathology
----------------------

Run tissue-level benchmarks with:

.. code-block:: bash

   python -m spora_bench.benchmarks.run_tissue_level_tasks \
     model_config=<path-to-model-config> \
     datasets_config=<path-to-datasets-config> \
     benchmarks_config=<path-to-benchmarks-config>

For example, to run VirTues on ``cords2024cancer``:

.. code-block:: bash

   python -m spora_bench.benchmarks.run_tissue_level_tasks \
     model_config=configs/models/virtues.yaml \
     datasets_config=configs/datasets/virtues/cords2024cancer.yaml \
     benchmarks_config=configs/benchmarks/tissue_level/cords2024cancer.yaml

Virtual staining and ASTIR
--------------------------

The repository also includes entrypoints for virtual staining and ASTIR-based
experiments. Use the corresponding modules under ``spora_bench.benchmarks``
with the same config-driven pattern as above.

Notes
-----

- Some models, such as KRONOS, require their own dataset config variants.
- If you want to run multiple benchmarks at once, wildcards are supported for
  some config arguments.

Supported models
-----------------

The benchmark pipeline currently supports the following model wrappers:

.. csv-table::
   :header: Model, Config, Supported tasks
   :widths: 15, 28, 40

   "VirTues", "``configs/models/virtues.yaml``", "Cell phenotyping, tissue-level pathology, virtual staining"
   "KRONOS", "``configs/models/kronos.yaml``", "Cell phenotyping, tissue-level pathology"
   "ResNet", "``configs/models/resnet.yaml``", "Tissue-level pathology only"
   "astir", "``configs/models/astir.yaml``", "Cell phenotyping only"

ASTIR uses a separate benchmark pipeline via
``spora_bench.benchmarks.run_astir`` and its configs live under
``configs/benchmarks/astir``.
