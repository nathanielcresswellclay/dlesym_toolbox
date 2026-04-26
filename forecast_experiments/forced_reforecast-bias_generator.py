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
    inference_args.output_directory = config.forecast_output_dir
    inference_args.data_directory = model_config.data.dst_directory
    inference_args.dataset_name = model_config.data.dataset_name if 'init_dataset' not in config else config.data.init_dataset
    inference_args.lead_time = config.leadtime # ~12month forecast in hours
    inference_args.hydra_path = os.path.relpath(config.model, os.path.join(os.getcwd(), 'hydra'))
    inference_args.gpu = 0

    # leave these as None for now
    inference_args.model_checkpoint = None
    inference_args.batch_size = None
    inference_args.encode_int = False
    inference_args.to_zarr = False
    inference_args.data_prefix = None
    inference_args.data_suffix = None
    inference_args.constant_dummy_scaling = False

    # this will hold all the forecast parameters 
    params = []
    # loop through time params and finish populating args
    for time_param in config.time_params:

        # make deep copy of args
        args_copy = OmegaConf.create(OmegaConf.to_container(inference_args, resolve=True))
        args_copy.forecast_init_start = time_param.init_start
        args_copy.forecast_init_end = time_param.init_end
        args_copy.freq = time_param.init_freq
        args_copy.output_filename = f"reforecast_{config.filename_suffix}_INIT-{time_param.init_start.replace('-','')}-{time_param.init_end.replace('-','')}_FREQ-{time_param.init_freq}"
        params.append(args_copy)

    # import the forecast function
    try:
        from inference.forecast_single_component import inference
    except ImportError as e:
        logger.error(f"Failed to import forecast_single_component: {e}")

    # loop through all the forecast parameters and run forecasts
    logger.info(f"Starting reforecasts for {len(params)} different initialization periods.")
    for i, inference_args in enumerate(params):

        # check if forecast already exists
        forecast_file = os.path.join(inference_args.output_directory, inference_args.output_filename + ".nc")
        if os.path.exists(forecast_file) and not config.overwrite_forecasts:
            logger.info(f"Forecast file {forecast_file} already exists for {inference_args.model_path} and overwrite_forecasts is set to False. Skipping forecast.")
            continue
        try:
            logger.info(f"Running forecast for model: {inference_args.model_path}")
            inference(inference_args)

        except Exception as e:
            logger.error(f"Failed to run forecast for {inference_args.model_path}: {e}")

        # log progress
        logger.info(f"Completed {i+1} out of {len(params)} forecasts.")

    # log completion
    logger.info("All forecasts completed successfully.")
    # close the log file
    log_file.close()


if __name__ == "__main__":

    # receive arguments 
    import argparse
    parser = argparse.ArgumentParser(description="Run reforecasts for bias calculation.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # run analyses 
    main(args.config)