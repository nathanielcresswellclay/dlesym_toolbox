import os
import sys
import time
import logging
from pprint import pprint
from omegaconf import OmegaConf
from toolbox_utils import setup_logging

def main(config_path: str):

    # load the config file
    config = OmegaConf.load(config_path)
    log_file, logger = setup_logging(config.output_log)


    logger.info("Loaded config:\n\n" + OmegaConf.to_yaml(config))  # prints the loaded YAML for clarity

    # get model config
    model_config = OmegaConf.load(os.path.join(config.model,".hydra/config.yaml"))

    # get config object 
    inference_args = OmegaConf.create()

    # populate arguments to forecast_single_component.py
    inference_args.model_path = config.model
    inference_args.output_directory = os.path.join(config.model, "forecasts")
    inference_args.output_filename = f"forecast_forced_{config.filename_suffix}"
    inference_args.data_directory = model_config.data.dst_directory
    inference_args.dataset_name = model_config.data.dataset_name if 'init_dataset' not in config else config.data.init_dataset
    inference_args.lead_time = 9216 # ~12month forecast in hours
    inference_args.forecast_init_start = config.init_time_start
    inference_args.forecast_init_end = config.init_time_end
    inference_args.hydra_path = os.path.relpath(config.model, os.path.join(os.getcwd(), 'hydra'))
    inference_args.freq = config.init_time_freq
    inference_args.gpu = 0
    # leave these as None for now
    inference_args.model_checkpoint = None
    inference_args.batch_size = None
    inference_args.encode_int = False
    inference_args.to_zarr = False
    inference_args.data_prefix = None
    inference_args.data_suffix = None
    inference_args.constant_dummy_scaling = False
    
    # import the forecast function
    try:
        from inference.forecast_single_component import inference
    except ImportError as e:
        logger.error(f"Failed to import forecast_single_component: {e}")

    # check to see if forecast file already exists
    forecast_file = os.path.join(inference_args.output_directory, inference_args.output_filename) + ".nc"

    # log the current inference tasks 
    if os.path.exists(forecast_file) and not config.overwrite_forecasts:
        logger.info(f"Forecast file {forecast_file} already exists for {inference_args.model_path} and overwrite_forecasts is set to False. Skipping forecast.")
    try:
        logger.info(f"Running forecast for model: {inference_args.model_path}")
        inference(inference_args)
    except Exception as e:
        logger.error(f"Failed to run forecast for {inference_args.model_path}: {e}")

    # log completion
    logger.info("All forecasts completed successfully.")
    # close the log file
    log_file.close()


if __name__ == "__main__":

    # receive arguments 
    import argparse
    parser = argparse.ArgumentParser(description="Run seasonal forecast.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # run analyses 
    main(args.config)