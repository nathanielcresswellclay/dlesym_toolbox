# Development Guide

Conventions for extending `dlesym_toolbox` with new inference and analysis routines.

## Pipeline Contract

`analysis_suite.py` launches subprocesses by module name:

- Inference module: `forecast_experiments/<experiment_name>.py`
- Analysis module: `<analysis_name>.py`

Each target script should accept one positional argument:

```bash
python <module>.py <config_path>
```

and expose a `main(config_path: str)` entry point.

## Required Script Pattern

For both inference and analysis scripts:

1. Load config with `OmegaConf.load(config_path)`.
2. Initialize logging (`toolbox_utils.setup_logging(config.output_log)`).
3. Perform work.
4. Exit cleanly from `if __name__ == "__main__":` argument parsing block.

## Writing a New Inference Module

Location:

- `forecast_experiments/<your_experiment_name>.py`

Recommended config keys:

- `output_log`
- `mounted_modules`
- `config_cache`
- experiment-specific model/data/runtime fields

Notes:

- `analysis_suite.py` sets `CUDA_VISIBLE_DEVICES` per launched inference subprocess.
- Use `overwrite` guards when generating forecast files to avoid accidental reruns.
- Keep output file naming deterministic so downstream analysis config can reference files.

## Writing a New Analysis Module

Location:

- top-level `<your_analysis_name>.py`

Recommended config keys:

- `output_log`
- `mounted_modules`
- `config_cache`
- `output_directory`
- paths to forecast and verification data

Optional:

- `conda_env` can be set in `analysis_params` to run this module under a specific environment.

## Adding a New End-to-End Workflow

1. Implement inference script (if required).
2. Implement analysis script.
3. Add an example YAML in `analysis_example_configs/`.
4. Run through `analysis_suite.py` with `run_inference` and/or `run_analysis`.
5. Add or update a batch script if this is commonly run on SLURM.

## Config Design Tips

- Use `${output_dir}` interpolation for portability.
- Store logs under per-analysis subdirectories.
- Keep one `config_cache` file per analysis item.
- Include all import roots in `mounted_modules` to prevent runtime import failures.

## Debugging and Reproducibility

- If a subprocess fails early, inspect its `output_log`.
- Prefer immutable output paths for published analyses and explicit overwrite flags for regeneration.

## Suggested Documentation Workflow

When adding a new routine, update:

- `CONFIG_REFERENCE.md` for new reusable config keys.
- example YAML in `analysis_example_configs/` for copy-paste discoverability.

