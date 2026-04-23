import os
import sys
import time
import traceback
import logging
import xarray as xr
import numpy as np
from pprint import pprint
from omegaconf import OmegaConf
from toolbox_utils import setup_logging


def main(config_path: str):

    # load the config file
    config = OmegaConf.load(config_path)
    log_file, logger = setup_logging(config.output_log)

    # initialize arg template
    # this template will be the base for calls to 
    # coupled_forecast_hdf5_3component.py. We're going to 
    # copy this template and add model-specific parameters 
    # so that we get a list corresponding to arguments that 
    # run appropriate forecasts for each model. 
    arg_template = OmegaConf.create()

    # add common arguments to template 
    arg_template.lead_time = config.leadtime # leadtime in hours 
    arg_template.forecast_init_start = config.forecast_init_start
    arg_template.forecast_init_end = config.forecast_init_end
    arg_template.freq = config.freq
    arg_template.gpu = 0
    arg_template.batch_size = None
    arg_template.encode_int = False
    arg_template.to_zarr = False
    arg_template.data_prefix = None
    arg_template.data_suffix = None
    arg_template.data_directory = None
    arg_template.time_chunk = 1 # chunk in initialization
    arg_template.step_chunk = 32 # include all steps
    arg_template.datetime = False
    # model and hydra paths won't change for each ensemble member
    arg_template.atmos_hydra_path = config.models.atmos.model_path
    arg_template.atmos_model_path = config.models.atmos.model_path
    arg_template.ocean_hydra_path = config.models.ocean.model_path
    arg_template.ocean_model_path = config.models.ocean.model_path
    arg_template.land_hydra_path = config.models.land.model_path
    arg_template.land_model_path = config.models.land.model_path
    # add model-specific dataset paths
    arg_template.atmos_dataset_path = config.models.atmos.atmos_dataset_path
    arg_template.ocean_dataset_path = config.models.ocean.ocean_dataset_path
    arg_template.land_dataset_path = config.models.land.land_dataset_path
    # output directory and cache directory will change for each ensemble member
    arg_template.output_directory = config.models.output_dir
    arg_template.cache_dir = config.models.output_dir

    # log the template
    logger.info(f"Template: {arg_template}")    

    # deduce ensemble members
    n_land_checkpoints = len(getattr(config.models.land, 'checkpoints', [1]))
    n_ocean_checkpoints = len(getattr(config.models.ocean, 'checkpoints', [1]))
    n_atmos_checkpoints = len(getattr(config.models.atmos, 'checkpoints', [1]))
    ensemble_members = max(n_land_checkpoints, n_ocean_checkpoints, n_atmos_checkpoints)

    # loop through models and add model-specific arguments
    inference_args = []
    land_forecasts = []
    ocean_forecasts = []
    atmos_forecasts = []
    for i in range(ensemble_members):   

        # get model configs
        atmos_model_config = OmegaConf.load(os.path.join(config.models.atmos.model_path,".hydra/config.yaml"))
        ocean_model_config = OmegaConf.load(os.path.join(config.models.ocean.model_path,".hydra/config.yaml"))
        land_model_config = OmegaConf.load(os.path.join(config.models.land.model_path,".hydra/config.yaml"))

        # deep copy the template
        arg = OmegaConf.create(arg_template)

        # add model-specific checkpoints and paths  
        arg.atmos_model_path = config.models.atmos.model_path
        if n_atmos_checkpoints > 1:
            arg.atmos_model_checkpoint = config.models.atmos.checkpoints[i]
        else:
            arg.atms_model_checkpoint = None
        arg.ocean_model_path = config.models.ocean.model_path
        if n_ocean_checkpoints > 1:
            arg.ocean_model_checkpoint = config.models.ocean.checkpoints[i]
        else:
            arg.ocean_model_checkpoint = None
        arg.ocean_model_path = config.models.ocean.model_path
        if n_land_checkpoints > 1:
            arg.land_model_checkpoint = config.models.land.checkpoints[i]
        else:
            arg.land_model_checkpoint = None

        arg.atmos_output_filename = f"atmos_{config.forecast_file_suffix}_n{i:02d}"
        arg.ocean_output_filename = f"ocean_{config.forecast_file_suffix}_n{i:02d}"
        arg.land_output_filename = f"land_{config.forecast_file_suffix}_n{i:02d}"

        # add model inference arguments to list
        inference_args.append(arg)

        # add forecasts to list
        atmos_forecasts.append(arg.atmos_output_filename)
        ocean_forecasts.append(arg.ocean_output_filename)
        land_forecasts.append(arg.land_output_filename)

        # clean up for next iteration
        del atmos_model_config
        del ocean_model_config
        del land_model_config
    
    # import the forecast function
    try:
        from inference.coupled_forecast_hdf5_3component import coupled_inference_hdf5
    except ImportError as e:
        logger.error(f"Failed to import coupled_forecast_hdf5_3component: {e}")

    # run the forecast for each model
    for i in range(ensemble_members):

        # check to see if forecast file already exists
        atmos_forecast_file = os.path.join(inference_args[i].output_directory, inference_args[i].atmos_output_filename) + ".nc"
        ocean_forecast_file = os.path.join(inference_args[i].output_directory, inference_args[i].ocean_output_filename) + ".nc"
        land_forecast_file = os.path.join(inference_args[i].output_directory, inference_args[i].land_output_filename) + ".nc"

        # log the current inference tasks 
        if os.path.exists(atmos_forecast_file) and os.path.exists(ocean_forecast_file) and os.path.exists(land_forecast_file) and not config.overwrite_forecasts:
            logger.info(f"Forecast files {atmos_forecast_file}, {ocean_forecast_file}, {land_forecast_file} already exist for {inference_args[i].output_directory} and overwrite_forecasts is set to False. Skipping forecast.")
            continue
        try:
            logger.info(f"Running forecast for model: {inference_args[i].output_directory}")
            coupled_inference_hdf5(inference_args[i])
        except Exception as e:
            logger.error(f"Failed to run forecast for {inference_args[i].output_directory}: {e}", exc_info=True)
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