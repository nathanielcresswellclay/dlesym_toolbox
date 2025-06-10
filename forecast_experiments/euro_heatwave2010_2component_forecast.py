import os
import sys
import time
import numpy as np
import traceback
import logging
from pprint import pprint
from omegaconf import OmegaConf
from toolbox_utils import setup_logging
from forecast_experiments.experiment_utils import resolve_output_directory

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
    arg_template.lead_time = 60*24 # leadtime in hours 
    arg_template.forecast_init_start = '2010-06-25T00:00:00' 
    arg_template.forecast_init_end = '2010-07-25T00:00:00'
    arg_template.freq = '1D'
    arg_template.gpu = 0
    arg_template.batch_size = None
    arg_template.time_chunk = 30         # for time and step chunks, we should
    arg_template.step_chunk = 60*24 / 4  # be able to fit this all in memory
    arg_template.encode_int = False
    arg_template.data_prefix = None
    arg_template.data_suffix = None
    arg_template.datetime = False # these ones aren't necessary at these leadtimes
    arg_template.end_date = None
    
    # here we resolved the coupling organization, basically the script, this this input flag,
    # organizes the atmos-ocean model pairs for the experiment. 
    atmos_ocean_pairs = []
    if config.coupling_organization == 'ordered':
        if len(config.atmos_models) != len(config.ocean_models):
            raise ValueError("When using 'ordered' coupling_organization, atmos_models and ocean_models must have the same length.")
        atmos_ocean_pairs = [(config.atmos_models[i], config.ocean_models[i]) for i in range(len(config.atmos_models))]
    elif config.coupling_organization == 'melange':
        for i in range(len(config.atmos_models)):
            for j in range(len(config.ocean_models)):
                atmos_ocean_pairs.append((config.atmos_models[i], config.ocean_models[j]))
    else:
        raise ValueError("Invalid coupling_organization. Must be 'ordered' or 'melange'.")

    # resolve which models to couple 
    # loop through models and add model-specific arguments
    inference_args = []
    for (atmos_model, ocean_model) in atmos_ocean_pairs:

        # get model hydra params
        atmos_model_config = OmegaConf.load(os.path.join(atmos_model,".hydra/config.yaml"))
        ocean_model_config = OmegaConf.load(os.path.join(ocean_model,".hydra/config.yaml"))

        # deep copy the template
        arg = OmegaConf.create(arg_template)

        # add model-specific arguments
        arg.atmos_model_path = atmos_model
        arg.ocean_model_path = ocean_model
        arg.atmos_model_checkpoint = None
        arg.ocean_model_checkpoint = None
        arg.atmos_output_filename = "atmos_forecast_coupled_euro-heatwave2010"
        arg.ocean_output_filename = "ocean_forecast_coupled_euro-heatwave2010"
        arg.output_directory = resolve_output_directory(atmos_model, ocean_model)
        # if the output directory does not exist, create it
        if not os.path.exists(arg.output_directory):
            logger.info(f"Creating output directory: {arg.output_directory}")
            os.makedirs(arg.output_directory, exist_ok=True)
        arg.cache_dir = os.path.join(arg.output_directory)
        # by setting the absolute path in init data
        # we allow for file in different directories
        # NOTE we also have to set data_directory to None
        # to avoid ambiguity with the data location in the
        # coupled forecast script
        arg.atmos_dataset_path = config.atmos_data_path
        arg.ocean_dataset_path = config.ocean_data_path
        arg.data_directory = None
        arg.dataset_name = None
        # hydra path is the relative path to the hydra directory
        arg.atmos_hydra_path = os.path.relpath(atmos_model, os.path.join(os.getcwd(), 'hydra'))
        arg.ocean_hydra_path = os.path.relpath(ocean_model, os.path.join(os.getcwd(), 'hydra'))
        
        # add model inference arguments to list
        inference_args.append(arg)

        # clean up 
        del atmos_model_config
        del ocean_model_config
    
    # import the forecast function
    logger.info("Importing coupled inference function...")
    try:
        from inference.coupled_forecast_hdf5 import coupled_inference_hdf5 as inference
    except ImportError as e:
        logger.error(f"Failed to import coupled_inference_hdf5: {e}")

    # run the forecast for each model
    for inference_arg in inference_args:

        # check to see if forecast files already exists
        atmos_forecast_file = os.path.join(inference_arg.output_directory, inference_arg.atmos_output_filename) + ".nc"
        ocean_forecast_file = os.path.join(inference_arg.output_directory, inference_arg.ocean_output_filename) + ".nc"

        # log the current inference tasks 
        if np.logical_and(os.path.exists(atmos_forecast_file), os.path.exists(ocean_forecast_file)) \
            and not config.overwrite_forecasts:
            logger.info(f"Forecast files {atmos_forecast_file} and {ocean_forecast_file} already exist for overwrite_forecasts is set to False. Skipping forecast.")
            continue
        try:
            logger.info(f"Running forecast for model pair: {inference_arg.atmos_model_path} and {inference_arg.ocean_model_path}")
            inference(inference_arg)
        except Exception as e:
            logger.error(f"Failed to run forecast for {inference_arg.atmos_model_path} and {inference_arg.ocean_model_path}:")
            traceback.print_exc()
            continue
    # log completion
    logger.info("All forecasts completed.")
    # close the log file
    log_file.close()


if __name__ == "__main__":

    # receive arguments 
    import argparse
    parser = argparse.ArgumentParser(description="Run coupled 2-component forecasts for 2010 european heatwave.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # run analyses 
    main(args.config)