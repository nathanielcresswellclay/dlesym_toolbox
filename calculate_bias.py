import os 
import xarray as xr 
import numpy as np
import pandas as pd
import time

import logging
from pprint import pprint
from omegaconf import OmegaConf
from toolbox_utils import setup_logging
from tqdm import tqdm


def main(config_path: str):

    # load the config file
    config = OmegaConf.load(config_path)
    log_file, logger = setup_logging(config.output_log)

    # check if output file already exists, if overwrite is False, then exit, otherwise continue
    if os.path.exists(config.output_filename) and not config.overwrite:
        logger.info(f"Output file {config.output_filename} already exists and overwrite is set to False. Exiting.")
        return
    elif os.path.exists(config.output_filename) and config.overwrite:
        logger.info(f"Output file {config.output_filename} already exists but overwrite is set to True. Continuing and overwriting the file.")
        os.remove(config.output_filename)

    # get list of files in reforecast directory 
    reforecast_files = sorted([os.path.join(config.reforecast_directory, f) for f in os.listdir(config.reforecast_directory) if f.endswith('.nc')])

    # open files as a single dataset
    logger.info(f"Opening {len(reforecast_files)} reforecast files from {config.reforecast_directory}. MAKE SURE ALL FILES IN DIRECTORY ARE REFORECASTS!")
    reforecasts = xr.open_mfdataset(reforecast_files, concat_dim='time', combine='nested')[config.variable]
    
    # load verification data
    logger.info(f"Loading verification data from {config.verification_file}")
    verification = xr.open_zarr(config.verification_file).targets.sel(channel_out=config.variable)

    # contruct verification array that matches reforecasts
    verif_forecast = xr.zeros_like(reforecasts)
    # attempt to load verification into memory
    logger.info("Loading verification data into memory...")
    timer = time.perf_counter()
    verification.load()
    logger.info(f"Verification data loaded into memory in {time.perf_counter() - timer:.2f} seconds.")
    # populate array 
    logger.info("Populating verification forecast...")
    timer = time.perf_counter()
    for t in tqdm(verif_forecast.time.values):
        for s in verif_forecast.step.values:
            valid_time = s + t 
            temp_data = verification.sel(time=valid_time).values
            verif_forecast.loc[dict(time=t, step=s)] = temp_data

    logger.info(f"Verification forecast populated in {time.perf_counter() - timer:.2f} seconds.")

    # calculate the error for forecasts
    error = reforecasts - verif_forecast
    # group by calendar day and average 
    init_calendar_day = error.time.dt.strftime('%m-%d')
    error = error.groupby(init_calendar_day).mean()

    # save to output file
    logger.info(f"Saving bias cache to {config.output_filename}")
    timer = time.perf_counter()
    error.to_netcdf(config.output_filename)
    logger.info(f"Bias cache saved successfully in {time.perf_counter() - timer:.2f} seconds.")
    print(error)

if __name__ == "__main__":

    # receive arguments 
    import argparse
    parser = argparse.ArgumentParser(description="Calculate bias from reforecasts.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # run analyses 
    main(args.config)