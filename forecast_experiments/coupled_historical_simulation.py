import os
import sys
import time
import logging
import pandas as pd
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
    
    # calculate forecast lead time in hours
    lead_time = int((pd.to_datetime(config.end_time) - pd.to_datetime(config.init_time)).total_seconds() / 3600)
    arg_template.lead_time = lead_time # leadtime in hours 
    arg_template.forecast_init_start =  config.init_time
    arg_template.atmos_model_checkpoint = None
    arg_template.ocean_model_checkpoint = None
    arg_template.land_model_checkpoint = None
    arg_template.forecast_init_end = str(pd.to_datetime(config.init_time)+pd.Timedelta(seconds=1))[:10]
    arg_template.freq = '1D'
    arg_template.gpu = 0
    arg_template.batch_size = None
    arg_template.encode_int = False
    arg_template.to_zarr = False
    arg_template.data_prefix = None
    arg_template.data_suffix = None
    arg_template.time_chunk = config.time_chunk # chunk in initialization
    arg_template.step_chunk = config.step_chunk # include all steps
    arg_template.datetime = False
    arg_template.data_directory = None

    # import the forecast function
    try:
        from inference.coupled_forecast_hdf5_3component import coupled_inference_hdf5
    except ImportError as e:
        logger.error(f"Failed to import forecast_single_component: {e}")

    # loop through models and add model-specific arguments
    inference_args = []
    for model_group in config.model_groups:

        # get model configs
        atmos_model_config = OmegaConf.load(os.path.join(model_group.atmos,".hydra/config.yaml"))
        ocean_model_config = OmegaConf.load(os.path.join(model_group.ocean,".hydra/config.yaml"))
        land_model_config = OmegaConf.load(os.path.join(model_group.land,".hydra/config.yaml"))
        # deep copy the template
        arg = OmegaConf.create(arg_template)
        # add model-specific arguments
        # add model-specific arguments
        arg.atmos_model_path = model_group.atmos
        arg.atmos_dataset_path = model_group.atmos_dataset_path
        arg.ocean_model_path = model_group.ocean
        arg.ocean_dataset_path = model_group.ocean_dataset_path
        arg.land_model_path = model_group.land
        arg.land_dataset_path = model_group.land_dataset_path
        arg.output_directory = model_group.output_dir
        arg.cache_dir = model_group.output_dir
        arg.atmos_output_filename = f"atmos_forecast-coupled_historical_{config.init_time.replace('-','.')}-{config.end_time.replace('-','.')}"
        arg.ocean_output_filename = f"ocean_forecast-coupled_historical_{config.init_time.replace('-','.')}-{config.end_time.replace('-','.')}"
        arg.land_output_filename = f"land_forecast-coupled_historical_{config.init_time.replace('-','.')}-{config.end_time.replace('-','.')}"
        arg.atmos_hydra_path = model_group.atmos
        arg.ocean_hydra_path = model_group.ocean
        arg.land_hydra_path = model_group.land

        # add model inference arguments to list
        inference_args.append(arg)

        # clean up model configs
        del atmos_model_config
        del ocean_model_config
        del land_model_config

    # run the forecast for each model
    for inference_arg in inference_args:
        if os.path.exists(os.path.join(inference_arg.output_directory, inference_arg.atmos_output_filename + ".nc")) and not config.overwrite_forecasts:
            logger.info(f"Forecast file {os.path.join(inference_arg.output_directory, inference_arg.atmos_output_filename + ".nc")} already exists for {inference_arg.output_directory} and overwrite_forecasts is set to False. Skipping forecast.")
            continue
        try:
            logger.info(f"Running forecast for model: {inference_arg.output_directory}")
            coupled_inference_hdf5(inference_arg)
        except Exception as e:
            logger.error(f"Failed to run forecast for {inference_arg.output_directory}: {e}", exc_info=True)
            continue
    # log completion
    logger.info("All forecasts completed.")
    # close the log file
    log_file.close()


if __name__ == "__main__":

    # receive arguments 
    import argparse
    parser = argparse.ArgumentParser(description="Run forecasts for 2003 european heatwave.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # run analyses 
    main(args.config)