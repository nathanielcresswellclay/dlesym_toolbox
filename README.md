# DLESyM Toolbox

Personal analysis and inference orchestration toolkit for evaluating ML Earth system model experiments.

This repository is organized around one entry point: `analysis_suite.py`. A single YAML config declares one or more analyses, each with:

- optional **inference** parameters (to generate forecast files first), and
- optional **analysis** parameters (to postprocess, score, and/or plot outputs).

The suite schedules inference jobs across available GPUs and analysis jobs across CPU subprocesses, so multi-experiment workflows can run with a single command.

## What This Repo Is For

- Run reproducible experiment pipelines from config files.
- Launch forecast/inference routines that depend on external model repos.
- Run analysis scripts against generated or existing forecasts.
- Keep logs, temporary run configs, and outputs organized per experiment.

## External Dependencies

This toolbox is designed to work with model/training/inference code that lives in separate repositories, commonly:

- [`NV-dlesm`](https://github.com/AtmosSci-DLESM/NV-dlesm/tree/ncresswell-dltm)
- [`modulus-uw`](https://github.com/AtmosSci-DLESM/modulus-uw/tree/ncc/dltm)
- [`DLESyM`](https://github.com/AtmosSci-DLESM/DLESyM)

Those paths are injected at runtime via each analysis block's `mounted_modules` setting (added to `PYTHONPATH` for subprocesses).

## Repository Layout

- `analysis_suite.py`: Pipeline orchestrator (inference phase -> analysis phase).
- `analysis_example_configs/`: Runnable YAML examples for common evaluations.
- `forecast_experiments/`: Inference launcher modules called by the suite.
- `*.py` at repo root: Analysis modules (climatology, anomaly diagnostics, skill metrics, etc.).
- `toolbox_utils.py`: Shared helpers (notably file-based logging setup).
- `batch_analysis_example/`: Example SLURM launch scripts.
- `notebooks/`: Debug/development notebooks.
- `analysis_example_outputs/`: Example/generated outputs and temporary config caches.

## Execution Model

`analysis_suite.py` expects a config with a top-level `analyses` list.

For each item in `analyses`:

1. `inference_params` (if present and `run_inference: True`) is sent to an inference module in `forecast_experiments/`.
2. `analysis_params` (if present and `run_analysis: True`) is sent to an analysis module (usually at repo root).

Parallelization:

- Inference: one subprocess per available GPU from `gpus`.
- Analysis: up to `n_proc` subprocesses.

Each subprocess receives a temporary resolved YAML (`config_cache`) written by the suite.

## Quick Start

If running on perlmutter, you will first need to get a GPU allocation interactively. Here's an example command: 

```bash
salloc --account=m4935  --image=registry.nersc.gov/m4935/dlesym-physicsnemo:25.06 --module=gpu,nccl-plugin --nodes 1 --ntasks-per-node 4 --constraint "gpu&hbm40g" --gpus-per-node 4 --qos interactive --time 02:00:00
```

Run from SLURM (example pattern):

```bash
shifter bash -c "cd /pscratch/sd/n/nacc/dlesym_toolbox && python analysis_suite.py /path/to/config.yaml"
```

For non-interactive use, see `batch_analysis_example/` for scheduler templates.

## Configuration

Use `analysis_example_configs/` as templates.

Top-level controls include:

- `output_dir`: root output directory for this run.
- `gpus`: GPU IDs available to inference subprocesses.
- `n_proc`: max parallel analysis subprocesses.
- `run_inference`: enable/disable inference phase.
- `run_analysis`: enable/disable analysis phase.
- `analyses`: list of analysis blocks.

A full field reference is documented in [`CONFIG_REFERENCE.md`](https://github.com/nathanielcresswellclay/dlesym_toolbox/blob/main/docs/CONFIG_REFERENCE.md).

## Logs and Outputs

- Most scripts write logs to `output_log` configured in each block.
- Analysis/inference subprocesses typically redirect stdout/stderr into those log files.
- Temporary config files (`config_cache`) are created per subprocess and removed by the suite after completion.

## Adding New Workflows

To add a new workflow:

1. Create a new inference module in `forecast_experiments/` (if needed) that is callable as a script and takes a string path to a configuration file.
2. Create a new analysis module (top-level `*.py`). This should also be callable as a script and take a string path to a configuration yaml.
3. Add a config in `analysis_example_configs/` wiring `experiment_name` and `analysis_name` as appropriate for the two routines above. 

Detailed conventions are in [`DEVELOPMENT_GUIDE.md`](https://github.com/nathanielcresswellclay/dlesym_toolbox/blob/main/docs/DEVELOPMENT_GUIDE.md).

