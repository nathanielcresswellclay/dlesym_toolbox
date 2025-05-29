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
    
    try: 
        from score.rmse_acc import plot_baseline_metrics
    except ImportError as e:
        logger.error("Failed to import required modules. Check mounted modules")
        logger.error(str(e))
        return
    
    # we will call scoring utility for each of the variables
    for var in config.variables:
        logger.info(f"Scoring variable: {var}")
        
        # populate forecast parameters for each model forecast
        # in our config forecast params we list some but not all
        # of the parameters that we need to pass to the scoring utility.
        forecast_params = []
        for forecast_param in config.forecast_params:
            # resolve and add missing parameters fore each model
            forecast_param.verification_file = config.verification_file # shared verification file
            # climatology file will be associated with verification file. Climos are saved variable-wise
            forecast_param.climatology_file = config.verification_file.replace('.zarr', f'_{var}-clima.nc') 
            # rmse and acc caches are assocated with forecast file, again saved variable-wise
            forecast_param.rmse_cache = forecast_param.file.replace('.nc', f'_{var}-rmse.nc')
            forecast_param.acc_cache = forecast_param.file.replace('.nc', f'_{var}-acc.nc')

            # add the forecast parameter to the list
            forecast_params.append(forecast_param)
        
        # run the scoring utility with configured params
        plot_baseline_metrics(
            forecast_params = forecast_params,
            variable = var,
            plot_file = config.output_directory + f"/skill_comp_{config.plot_file_suffix}_{var}.png",
            xlim = config.xlim,
            overwrite_cache = config.overwrite_cache,
        )
    
    logger.info("Skill score comparison completed successfully.")



if __name__ == "__main__":

    # receive arguments 
    import argparse
    parser = argparse.ArgumentParser(description="Run skill score comparison between forecasts.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # run analyses 
    main(args.config)