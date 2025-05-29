import os
import sys
import time
import logging

import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import OmegaConf
from toolbox_utils import setup_logging

def main(config_path: str):

    # load the config file
    config = OmegaConf.load(config_path)
    log_file, logger = setup_logging(config.output_log)

    logger.info("Loaded config:\n\n" + OmegaConf.to_yaml(config))  # prints the loaded YAML for clarity
    
    # try to imoport remap module from DLESyM 
    # NOTE there are some brittle dependencies to get remap to work properly
    # for environment recipe: https://github.com/AtmosSci-DLESM/DLESyM/blob/main/environments/dlesym-0.1.yaml
    try: 
        import data_processing.remap.healpix as hpx 
    except ImportError as e:
        logger.error("Failed to import reamp data_processing.remap.healpix")
        logger.error(str(e))
        return  
    
    # load our verification data and select t2m during the heatwave
    verif_temp = xr.open_zarr(config.verification_file).targets.sel(
        time=pd.date_range('2021-06-25', '2021-07-01', freq='6H'),
        channel_out="t2m0"
    )
    logger.info(f'verif_temp.values[:,6,30,30]: {verif_temp.values[:,6,30,30]}')
    # we're calculating daily anomalies, so we need to average over the 6-hourly data
    # then reformat the output of groupby to have useful time coordinates
    verif_temp = verif_temp.groupby("time.day").mean("time").rename({"day": "time"}).assign_coords(
        time=pd.date_range('2021-06-25', '2021-07-01', freq='D')
    )
    # next we need to get the climatology for the same period. If we already have 
    # a climaology file cached, we can load it, otherwise we need to calculate it
    climatology_file = config.verification_file.replace('.zarr', '_t2m0-clima.nc')
    if os.path.exists(climatology_file):
        logger.info(f"Loading cached climatology from {climatology_file}")
        climatology = xr.open_dataset(climatology_file).targets
    else:
        logger.info("Calculating climatology for the heatwave period")
        climatology = xr.open_zarr(config.verification_file).targets.sel(
            channel_out="t2m0"
        ).groupby('time.dayofyear').mean(dim='time')
        climatology.to_netcdf(climatology_file)
        logger.info(f"Climatology saved to {climatology_file}")
    # select heatwave period from climatology. climatology time dim is dayofyear
    # so we need to select the days corresponding to the heatwave, then fix dims
    climatology = climatology.sel(
        dayofyear=pd.date_range('2021-06-25', '2021-07-01', freq='D').dayofyear
    ).rename({"dayofyear": "time"}).assign_coords(
        time=pd.date_range('2021-06-25', '2021-07-01', freq='D')
    )

    # now we calculate the observed anomalies, and average over time
    logger.info(f'verif_temp.values.flatten()[0:10]: {verif_temp.values.flatten()[0:10]}')
    logger.info(f'climatology.values.flatten()[0:10]: {climatology.values.flatten()[0:10]}')
    verif_temp_anomaly = verif_temp - climatology
    verif_temp_anomaly = verif_temp_anomaly.mean(dim='time').reset_coords(drop=True)
    # we also need to plot on lat lon grid, so we need to remap the data,
    # this remapper will be used for forecasts as well. Use 1 degree resolution
    mapper = hpx.HEALPixRemap(
        latitudes=181,
        longitudes=360,
        nside=64,
    )
    # remap the verification data to lat lon grid
    verif_temp_anomaly_ll = xr.DataArray(
        mapper.hpx2ll(verif_temp_anomaly.values),
        dims=['lat', 'lon'], 
        coords={'lat': np.arange(-90, 90.1, 1),'lon': np.arange(0, 360, 1)}
    )

    # now we loop through to given forecasts and for each one calculate and plot
    # the anomalies during the heatwave period forecasted by each initiailzation
    # verification file is the same for all forecasts. Plots are saved to the 
    # indicated output directory
    # for forecast in config.forecast_params: 

    logger.info(f'verif_temp_anomaly.values.flatten()[0:10]: {verif_temp_anomaly.values.flatten()[0:10]}')
    logger.info(f'verif_temp_anomaly_ll.values.flatten()[0:10]: {verif_temp_anomaly_ll.values.flatten()[0:10]}')
    logger.info("Heatwave anomaly analysis completed successfully.")



if __name__ == "__main__":

    # receive arguments 
    import argparse
    parser = argparse.ArgumentParser(description="Run skill score comparison between forecasts.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # run analyses 
    main(args.config)