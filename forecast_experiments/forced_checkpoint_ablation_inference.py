import os
import sys
import time
import logging
from pprint import pprint
from omegaconf import OmegaConf
from toolbox_utils import setup_logging

def get_checkpoints(model_path: str):

    path_to_checkpoints = os.path.join(model_path, "tensorboard", "checkpoints")
    return [f for f in os.listdir(path_to_checkpoints) if f.endswith('.mdlus')]

def main(config_path: str):

    # load the config file
    config = OmegaConf.load(config_path)
    log_file, logger = setup_logging(config.output_log)

    # initialize arg template
    # this template will be the base for calls to 
    # forecast_single_component.py. We're going to 
    # copy this template and add model-specific parameters 
    # so that we get a list corresponding to arguments that 
    # run appropriate forecasts for each model. 
    arg_template = OmegaConf.create()

    # create directory to hold checkpoint ablation forecasts model/forecasts/forced_simulation_checkpoint_ablation
    output_dir = os.path.join(config.model, "forecasts", "60day_forced_checkpoint_ablation")
    os.makedirs(output_dir, exist_ok=True)

    # add common arguments to template 
    model_config = OmegaConf.load(os.path.join(config.model,".hydra/config.yaml"))
    arg_template.model_path = config.model
    arg_template.output_directory = output_dir
    arg_template.data_directory = model_config.data.dst_directory
    arg_template.dataset_name = model_config.data.dataset_name
    arg_template.hydra_path = os.path.relpath(config.model, os.path.join(os.getcwd(), 'hydra'))
    arg_template.batch_size = None  # None will use default batch size
    arg_template.encode_int = False  # do not encode integers
    arg_template.to_zarr = False  # do not convert to zarr format
    arg_template.data_prefix = None  # no prefix for data files
    arg_template.data_suffix = None  # no suffix for data files
    arg_template.gpu = 0  # gpu visibility handled by client routine
    arg_template.constant_dummy_scaling = False  # do not use constant dummy scaling
    arg_template.model_checkpoint = None  # None will use all checkpoints
    arg_template.lead_time = 60*24 # leadtime in hours 
    arg_template.forecast_init_start = config.forecast_init_start
    arg_template.forecast_init_end = config.forecast_init_end
    arg_template.freq = 'biweekly'
    arg_template.gpu = 0

    if config.checkpoints is not None:
        # if checkpoints are specified use those
        checkpoints = config.checkpoints
    else:
        # if no checkpoints are specified, use all checkpoints
        checkpoints = get_checkpoints(config.model)

    # compile inference parameters for each checkpoint
    logger.info(f"Performing 60-day forced forecast comparison for checkpoints: {checkpoints}")
    inference_args = []
    for ckpt in checkpoints:

        # add model inference arguments to list
        arg = OmegaConf.create(arg_template)
        arg.model_checkpoint = os.path.join(arg.model_path, "tensorboard", "checkpoints", ckpt)
        arg.output_filename = f"forecast_forced_60day_{ckpt.replace('.mdlus', '')}" # remove the .mdlus extension
        inference_args.append(arg)
    
    # import the forecast function
    try:
        from inference.forecast_single_component import inference
    except ImportError as e:
        logger.error(f"Failed to import forecast_single_component: {e}")

    # run the forecast for each model
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
            logger.error(f"Failed to run forecast for {arg.model_path}: {e}")
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