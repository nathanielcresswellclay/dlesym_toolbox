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

    # initialize arg template
    # this template will be the base for calls to 
    # forecast_single_component.py. We're going to 
    # copy this template and add model-specific parameters 
    # so that we get a list corresponding to arguments that 
    # run appropriate forecasts for each model. 
    arg_template = OmegaConf.create()

    # add common arguments to template 
    arg_template.lead_time = 30*24 # leadtime in hours 
    arg_template.forecast_init_start = '2021-06-01T00:00:00' 
    arg_template.forecast_init_end = '2021-06-17T00:00:00'
    arg_template.freq = '1D'
    arg_template.gpu = 0

    # loop through models and add model-specific arguments
    inference_args = []
    for model in config.models:

        # get model config
        model_config = OmegaConf.load(os.path.join(model,".hydra/config.yaml"))
        # deep copy the template
        arg = OmegaConf.create(arg_template)
        # add model-specific arguments
        arg.model_path = model
        arg.output_directory = os.path.join(model, "forecasts")
        arg.output_filename = "forecast_forced_pnw-heatwave2021"
        arg.data_directory = model_config.data.dst_directory
        arg.dataset_name = model_config.data.dataset_name
        arg.hydra_path = os.path.relpath(model, os.path.join(os.getcwd(), 'hydra'))
        arg.model_checkpoint = None
        arg.batch_size = None
        arg.encode_int = False
        arg.to_zarr = False
        arg.data_prefix = None
        arg.data_suffix = None
        arg.constant_dummy_scaling = False

        # add model inference arguments to list
        inference_args.append(arg)

        # clean up 
        del model_config
    
    # import the forecast function
    try:
        from inference.forecast_single_component import inference
    except ImportError as e:
        logger.error(f"Failed to import forecast_single_component: {e}")

    # run the forecast for each model
    for inference_arg in inference_args:

        # check to see if forecast file already exists
        forecast_file = os.path.join(inference_arg.output_directory, inference_arg.output_filename) + ".nc"

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
    parser = argparse.ArgumentParser(description="Run forecasts for 2021 pnw heatwave.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # run analyses 
    main(args.config)