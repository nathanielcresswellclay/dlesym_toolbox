import os
import re
import sys
import time
import glob
import logging
from pprint import pprint
from omegaconf import OmegaConf
from toolbox_utils import setup_logging


def get_checkpoints(model_path: str):
    """Return all .mdlus checkpoint filenames in the model's checkpoint directory."""
    path_to_checkpoints = os.path.join(model_path, "tensorboard", "checkpoints")
    return [f for f in os.listdir(path_to_checkpoints) if f.endswith('.mdlus')]


def get_n_best_checkpoints(model_path: str, n: int):
    """
    Return the n best checkpoint filenames by validation loss (ascending).
    Only considers training-state-epoch-*.mdlus files with parseable val_loss.
    """
    path = os.path.join(model_path, "tensorboard", "checkpoints")
    path = os.path.abspath(path)
    ckpt_paths = glob.glob(os.path.join(path, "training-state-epoch-*.mdlus"))
    ranked = []
    for ckpt_path in ckpt_paths:
        if "NAN" in ckpt_path:
            continue
        match = re.findall(r"-?\d*\.?\d+E[+-]?\d+", os.path.basename(ckpt_path))
        if not match:
            continue
        curr_error = float(match[0])
        ranked.append((curr_error, os.path.basename(ckpt_path)))
    ranked.sort(key=lambda x: x[0])
    return [name for _, name in ranked[:n]]


def main(config_path: str):

    # load the config file
    config = OmegaConf.load(config_path)
    # merge defaults (preserves existing behavior when omitted)
    config_defaults = OmegaConf.create({
        'lead_time': 60 * 24,
        'forecast_init_start': '2016-06-30',
        'forecast_init_end': '2017-05-01',
        'freq': 'biweekly',
        'output_filename_prefix': 'forecast_forced_60day_',
    })
    config = OmegaConf.merge(config_defaults, config)

    log_file, logger = setup_logging(config.output_log)

    try:
        from inference.forecast_single_component import inference
        import inference.forecast_single_component as inference_module
        inference_module_dir = os.path.dirname(os.path.abspath(inference_module.__file__))
    except ImportError as e:
        logger.error(f"Failed to import forecast_single_component: {e}")
        return

    # initialize arg template
    arg_template = OmegaConf.create()

    # create directory to hold checkpoint ablation forecasts
    output_dir = os.path.join(config.model, "forecasts", f"{config.lead_time}h_forced_checkpoint_ablation")
    os.makedirs(output_dir, exist_ok=True)

    # add common arguments to template
    model_config = OmegaConf.load(os.path.join(config.model, ".hydra/config.yaml"))
    arg_template.model_path = config.model
    arg_template.output_directory = output_dir
    arg_template.data_directory = model_config.data.dst_directory
    arg_template.dataset_name = (
        model_config.data.dataset_name
        if 'init_dataset' not in config
        else config.init_dataset
    )
    arg_template.hydra_path = os.path.relpath(os.path.abspath(config.model), inference_module_dir)
    arg_template.batch_size = None
    arg_template.encode_int = False
    arg_template.to_zarr = False
    arg_template.data_prefix = None
    arg_template.data_suffix = None
    arg_template.gpu = 0
    arg_template.constant_dummy_scaling = False
    arg_template.model_checkpoint = None
    arg_template.lead_time = config.lead_time
    arg_template.forecast_init_start = config.forecast_init_start
    arg_template.forecast_init_end = config.forecast_init_end
    arg_template.freq = config.freq

    # checkpoint selection: explicit list > n_checkpoints > all .mdlus
    if config.get('checkpoints') is not None and len(config.checkpoints) > 0:
        checkpoints = list(config.checkpoints)
    elif config.get('n_checkpoints') is not None:
        checkpoints = get_n_best_checkpoints(config.model, config.n_checkpoints)
        logger.info(f"Selected {len(checkpoints)} best checkpoints by val_loss: {checkpoints}")
    else:
        checkpoints = get_checkpoints(config.model)

    # compile inference parameters for each checkpoint
    logger.info(f"Performing forced forecast comparison for checkpoints: {checkpoints}")
    inference_args = []
    for ckpt in checkpoints:
        arg = OmegaConf.create(arg_template)
        arg.model_checkpoint = ckpt  # filename only; forecast_single_component joins with checkpoint dir
        arg.output_filename = f"{config.output_filename_prefix}{ckpt.replace('.mdlus', '')}"
        inference_args.append(arg)

    # run the forecast for each checkpoint
    for inference_arg in inference_args:

        # check to see if forecast file already exists
        forecast_file = os.path.join(inference_arg.output_directory, inference_arg.output_filename) + ".nc"
        logger.info(f"Running forecast for {inference_arg.model_path} with checkpoint {inference_arg.model_checkpoint}")
        # log the current inference tasks 
        if os.path.exists(forecast_file) and not config.overwrite_forecasts:
            logger.info(f"Forecast file {forecast_file} already exists for {inference_arg.model_path} and overwrite_forecasts is set to False. Skipping forecast.")
            continue
        try:
            logger.info(f"Running forecast for model: {inference_arg.model_path}")
            inference(inference_arg)
        except Exception as e:
            logger.error(f"Failed to run forecast for {inference_arg.model_path}: {e}")
            continue  
   # log completion
    logger.info("All forecasts completed successfully.")
    # close the log file
    log_file.close()


if __name__ == "__main__":

    # receive arguments 
    import argparse
    parser = argparse.ArgumentParser(description="Run 60day forced forecast comparison.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # run analyses 
    main(args.config)