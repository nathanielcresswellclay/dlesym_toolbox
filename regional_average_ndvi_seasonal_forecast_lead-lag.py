import os
import sys
import time
import logging
import warnings

# analysis
import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import OmegaConf, DictConfig

# plotting 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
from cartopy.util import add_cyclic_point
import cartopy.io.img_tiles as cimgt

# mounted modules 
from toolbox_utils import setup_logging

def main(config_path: str):

    """
    send config to routines that plot t2m and z500 anomalies during the 2003 heatwave in Europe

    """

    # load the config file
    config = OmegaConf.load(config_path)
    _, logger = setup_logging(config.output_log)

    logger.info("Loaded config:\n\n" + OmegaConf.to_yaml(config))  # prints the loaded YAML for clarity

    _plot_ndvi_seasonal_forecasts(config, logger)
    logger.info("Finished plotting NDVI seasonal forecasts.")

def _plot_ndvi_seasonal_forecasts(config: DictConfig, logger: logging.Logger):

    # try to import remap module from DLESyM
    # NOTE there are some brittle dependencies to get remap to work properly
    # for environment recipe: https://github.com/AtmosSci-DLESM/DLESyM/blob/main/environments/dlesym-0.1.yaml
    try: 
        import data_processing.remap.healpix as hpx 
    except ImportError as e:
        logger.error("Failed to import reamp data_processing.remap.healpix")
        logger.error(str(e))
        return  

    # if plot climatology is set to True, we will calculate and plot the climatology
    if config.plot_climatology:

        # calculate climatology if it does not exist
        climatology_file = config.verification_file.replace('.zarr', '_ndvi-clima.nc')
        if os.path.exists(climatology_file):
            logger.info(f"Loading cached climatology from {climatology_file}")
            climatology = xr.open_dataset(climatology_file).targets
        else:
            logger.info("Calculating climatology NDVI_gapfill from verification data")
            climatology = xr.open_zarr(config.verification_file).targets.sel(
                channel_out="NDVI_gapfill",
            ).groupby('time.dayofyear').mean(dim='time')
            climatology.to_netcdf(climatology_file)
            logger.info(f"Climatology saved to {climatology_file}")
        
    # get indexing array for specified region 
    lat = xr.open_zarr(config.verification_file).lat 
    lon = xr.open_zarr(config.verification_file).lon
    # fix lon to support negative values
    lon = (lon + 180) % 360 - 180
    # create boolean index for the region box
    region_index_lat =  np.logical_and(lat > config.region_box.south , lat < config.region_box.north)
    region_index_lon = np.logical_and(lon > config.region_box.west , lon < config.region_box.east)
    region_index = region_index_lat & region_index_lon
    # replace false with nan to index the region


    # if indicated to visualize the region box, plot it
    if config.plot_region_map:
        logger.info("Plotting region box")
        _plot_region_box(config, save_path=os.path.join(config.output_directory, 'region_box.png'))

    # open_forecast 
    logger.info(f"Opening forecast file {config.forecast}")
    fcst = xr.open_dataset(config.forecast).NDVI_gapfill
    # apply bias correction if bias cache is provided
    if getattr(config, 'bias_cache', None) is not None:
        logger.info(f"Applying bias correction using bias cache {config.bias_cache}")
        bias = xr.open_dataset(config.bias_cache).NDVI_gapfill
        # align the bias dimensions with the forecast dimensions
        init_calendar_day = fcst.time.dt.strftime('%m-%d')
        fcst = fcst.assign_coords({'strftime': init_calendar_day})
        fcst = fcst - bias
        # apply the bias correction

    # select the region and average over it
    fcst = fcst.where(region_index.compute(), drop=True).mean(dim=['face', 'height', 'width'])

    # plot verification if indicated
    logger.info('Calculating regional average NDVI from verification data')
    obs = xr.open_zarr(config.verification_file).targets.sel(channel_out='NDVI_gapfill').sel(time=slice(config.plot_time_start, config.plot_time_end))
    obs = obs.where(region_index.compute(), drop=True).mean(dim=['face', 'height', 'width'])
    # calculate weekly average 
    obs = obs.resample(time='7D').mean()

    # initialize figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(config.figure_params.title, fontsize=16)

    # plot observed data
    logger.info("Plotting observed NDVI data")
    ax.plot(obs.time, obs, label='MODIS Observations', color='black', linewidth=2)

    # check if ensemble mean or individual members are to be plotted
    if config.plot_ensemble_mean:
        logger.info("Plotting ensemble mean of forecasts")
        fcsts = []
    else: 
        logger.info("Plotting individual ensemble members of forecasts")

    # loop through lead-lag ensemble and plot 
    for i, init in enumerate(fcst.time.values):
        # resolve valid time
        valid_time = init + fcst.step.values
        # select the data for the current initialization time
        fcst_init = fcst.sel(time=init)
        # fix step dimension to valid time
        fcst_init = fcst_init.assign_coords({'step': valid_time}).squeeze()
        # select indicated times to plot
        fcst_init = fcst_init.sel(step=slice(config.plot_time_start, config.plot_time_end))
        # weekly average
        fcst_init = fcst_init.resample(step='7D').mean()
        
        # if plotting ensemble mean, append. Otherwise, plot directly
        if config.plot_ensemble_mean:
            fcsts.append(fcst_init)
        else:
            # plot the data
            ax.plot(fcst_init.step, fcst_init, 
                    label='DLTM Ensemble' if i==0 else None, 
                    linewidth=1, color='lightseagreen')
    
    # if plotting ensemble mean, calculate and plot it
    if config.plot_ensemble_mean:
        logger.info("Calculating and plotting ensemble mean of forecasts")
        fcsts = xr.concat(fcsts, dim='time')
        fcst_mean = fcsts.mean(dim='time')
        ax.plot(fcst_mean.step, fcst_mean, label='DLTM Ensemble Mean', color='lightseagreen', linewidth=2)
    
    # style the plot
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('NDVI', fontsize=14)
    ax.legend()

    # save figure 
    output_file = os.path.join(config.output_directory, 'regional_average_ndvi.png')
    logger.info(f"Saving regional average NDVI plot to {output_file}")
    fig.savefig(output_file, **config.figure_params.savefig_params)
    
def _plot_region_box(config, save_path=None):
    """
    Plot a world map with continents/oceans filled, and draw a box for the region
    defined in config.region_box.
    """

    # read bounds from config
    west = config.region_box.west
    east = config.region_box.east
    south = config.region_box.south
    north = config.region_box.north

    # setup figure
    fig, ax = plt.subplots(
        figsize=(10, 5),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # add features
    ax.add_feature(cfeature.LAND, facecolor="darkseagreen")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    # ax.stock_img()  # use stock image for better aesthetics
    ax.add_feature(cfeature.LAKES, facecolor="lightblue", edgecolor="black", linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)

    # set global extent
    ax.set_global()
    ax.gridlines(draw_labels=True, linewidth=0.5, linestyle="--")

    # add rectangle for region box
    rect = Rectangle(
        (west, south),             # lower-left corner
        east - west,               # width
        north - south,             # height
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        transform=ccrs.PlateCarree(),
        zorder=10
    )
    ax.add_patch(rect)

    # optional zoom to region
    ax.set_extent([west - 10, east + 10, south - 10, north + 10], crs=ccrs.PlateCarree())

    # save or show
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    else:
        plt.show()

if __name__ == "__main__":

    # receive arguments 
    import argparse
    parser = argparse.ArgumentParser(description="Plot average NDVI over seasonal forecast assuming lead-lag ensemble.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # run analyses 
    main(args.config)