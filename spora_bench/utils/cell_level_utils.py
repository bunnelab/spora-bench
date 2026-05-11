def config_has_cell_level_benchmark(config):
    if "benchmarks" not in config:
        return False
    if "cell_level" not in config.benchmarks:
        return False
    return True