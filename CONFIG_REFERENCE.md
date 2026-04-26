# Analysis Suite Config Reference

This document describes the YAML schema expected by `analysis_suite.py`.

## Top-Level Fields

```yaml
experiment_name: <string>
toolbox_dir: <path>
output_dir: <path>
gpus: [0, 1, ...]
n_proc: <int>
run_inference: <bool>
run_analysis: <bool>
analyses: [...]
```

- `experiment_name`: Human-readable label for the run.
- `toolbox_dir`: Path to this repository (optional in code, useful for clarity).
- `output_dir`: Root output directory for the run.
- `gpus`: GPU IDs used to schedule inference subprocesses.
- `n_proc`: Maximum concurrent analysis subprocesses.
- `run_inference`: If true, run inference phase.
- `run_analysis`: If true, run analysis phase.
- `analyses`: List of analysis definitions.

## `analyses` Item Schema

Each item can define both phases:

```yaml
analyses:
  - name: <string>
    output_subdir: <path>
    inference_params: {...}   # optional
    analysis_params: {...}    # optional
```

- `name`: Label for readability.
- `output_subdir`: Directory created before execution.
- `inference_params`: Passed to an inference script in `forecast_experiments/`.
- `analysis_params`: Passed to an analysis script.

## Inference Parameters

Minimum common fields:

```yaml
inference_params:
  experiment_name: <module_name_without_py>
  output_log: <path_to_log_file>
  mounted_modules:
    - /path/to/repo_or_module
  config_cache: <temp_yaml_path>
```

- `experiment_name`: Selects script `forecast_experiments/<experiment_name>.py`.
- `output_log`: Log destination used by many inference scripts.
- `mounted_modules`: Prepended to `PYTHONPATH` in subprocess.
- `config_cache`: Temporary resolved YAML written by suite and passed to the subprocess.

Additional fields are experiment-specific. Common examples include:

- model paths/checkpoints
- dataset paths
- start/end times
- chunk settings
- overwrite toggles

## Analysis Parameters

Minimum common fields:

```yaml
analysis_params:
  analysis_name: <module_name_without_py>
  output_log: <path_to_log_file>
  mounted_modules:
    - /path/to/repo_or_module
  config_cache: <temp_yaml_path>
```

- `analysis_name`: Selects script `<analysis_name>.py`.
- `output_log`: Log destination used by analysis scripts.
- `mounted_modules`: Prepended to `PYTHONPATH` in subprocess.
- `config_cache`: Temporary resolved YAML written by suite and passed to the subprocess.

Optional:

```yaml
analysis_params:
  conda_env: /path/to/conda/env
```

If `conda_env` is present, suite invokes `<conda_env>/bin/python` for that analysis subprocess.

## Interpolation and Resolved Configs

YAML interpolations (for example `${output_dir}`) are resolved by `OmegaConf` before writing each `config_cache` file.

This makes subprocess configs self-contained and explicit.

## Running One Phase Only

Inference only:

```yaml
run_inference: true
run_analysis: false
```

Analysis only (using existing forecast files):

```yaml
run_inference: false
run_analysis: true
```

## Practical Notes

- Keep `config_cache` unique per analysis block to avoid collisions.
- Point `output_log` into each block's `output_subdir` for easier debugging.
- Ensure `mounted_modules` includes all external repos your target script imports.
- Where possible use conda environments for analysis as mounted modules are not actively maintained. 
- Inference scripts run from repository root and are referenced via `forecast_experiments/<name>.py`.

## Examples

See:

- `analysis_example_configs/coupled_climo_dev.yaml`
- `analysis_example_configs/skill_score_comparison.yaml`
- other configs in `analysis_example_configs/`

