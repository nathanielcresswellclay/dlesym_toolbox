import os
import sys
import time
import logging
import warnings
from dataclasses import dataclass

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
from cartopy.util import add_cyclic_point

# mounted modules 
from toolbox_utils import setup_logging

# useful for manipulating omega conf configs 
@dataclass
class ForecastParam:
    file: str
    model_id: str

def main(config_path: str):

    """
    send config to routines that plot t2m and z500 anomalies during the 2003 heatwave in Europe

    """

    # load the config file
    config = OmegaConf.load(config_path)
    _, logger = setup_logging(config.output_log)

    logger.info("Loaded config:\n\n" + OmegaConf.to_yaml(config))  # prints the loaded YAML for clarity
    
    # formats config to run with _plot_ndvi_climo
    config.forecast_params = _prepare_forecast_params(config)

    _plot_ndvi_climo(config, logger)
    logger.info("Finished plotting climatology for NDVI.")

def _prepare_forecast_params(config: DictConfig):

    # helper func to get epoch
    def _get_epoch(filename: str):
        return filename.split('-')[-3]  # get the second to last part of the path

    # get list of filepaths in the forecast directory
    forecast_files = [
        os.path.join(config.forecast_directory, f)
        for f in os.listdir(config.forecast_directory)
        if f.endswith('.nc') and 'forecast_forced' in f
    ]
    # get file IDs from the filenames
    forecast_ids = [
        f"epoch-{_get_epoch(f)}"
        for f in forecast_files
    ]
    return [
        ForecastParam(file=fcst_file, model_id=fcst_id)
        for fcst_file, fcst_id in zip(forecast_files, forecast_ids)
    ]

# colormap with white as the middle color, coolwarm base
def _get_custom_cmap_brbg():
    """
    Create a custom colormap with white as the first color.
    """
    # Get the 'coolwarm' colormap
    bwr = cm.get_cmap('BrBG')
    # Create a new colormap with white as the middle color
    new_colors = bwr(np.linspace(0, 1, 256))
    new_cmap = mcolors.ListedColormap(new_colors)
    return new_cmap

def _plot_global(data: xr.DataArray, mapper, output_file_prefix: str):
    """
    Plot global data on a map using the specified mapper.
    """
    for season in data.season.values:
        # select the data for the current season
        season_data = data.sel(season=season)
        # remap the data to lat/lon grid
        data_ll = xr.DataArray(
            mapper.hpx2ll(season_data.values),
            dims=['lat', 'lon'], 
            coords={'lat': np.arange(90, -90.1, -1),'lon': np.arange(0, 360, 1)}
        )

        fig, ax = plt.subplots(
            figsize=(10, 5), subplot_kw={'projection': ccrs.Robinson()}
        )
        ax.set_title(f"{season} Climatology", fontsize=15)
        ax.add_feature(cfeature.OCEAN, zorder=10, facecolor='white')
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax.coastlines()

        # add cyclic point to the data for plotting
        data_ll_cyclic, lon_cyclic = add_cyclic_point(
            data_ll, coord=data_ll.lon
        )
        
        # plot the data
        im = ax.contourf(
            lon_cyclic, data_ll.lat, data_ll_cyclic,
            transform=ccrs.PlateCarree(),
            cmap=_get_custom_cmap_brbg(),
            levels=np.arange(-.25, 1.0, 0.05),
            extend='both',
        )

        # add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
        cbar.set_label('NDVI', fontsize=12)
        # title and save the figure
        plt.title(f"{season}", fontsize=15)
        plt.savefig(f"{output_file_prefix}_{season}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    return

def _plot_ndvi_climo(config: DictConfig , logger: logging.Logger):

    # try to import remap module from DLESyM
    # NOTE there are some brittle dependencies to get remap to work properly
    # for environment recipe: https://github.com/AtmosSci-DLESM/DLESyM/blob/main/environments/dlesym-0.1.yaml
    try: 
        import data_processing.remap.healpix as hpx 
    except ImportError as e:
        logger.error("Failed to import reamp data_processing.remap.healpix")
        logger.error(str(e))
        return  

    # next we need to get the climatology of verif data for comparison
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
    # average climo into seasons: DJF, MAM, JJA, SON from day of year values
    climatology = climatology.assign_coords(dayofyear = pd.date_range('2000-01-01', '2000-12-31', freq='D')).groupby('dayofyear.season').mean(dim='dayofyear')

    # we also need to plot on lat lon grid, so we need to remap the data,
    # this remapper will be used for forecasts as well. Use 1 degree resolution
    mapper = hpx.HEALPixRemap(
        latitudes=181,
        longitudes=360,
        nside=64,
    )

    # plot seasons from observed climatology
    obs_climatology_file_prefix = config.output_directory + '/obs_climatology'
    logger.info(f"Plotting observed climatology to {obs_climatology_file_prefix}*")
    _plot_global(climatology, mapper, obs_climatology_file_prefix)

    # now we loop through to given forecasts and for each one calculate and plot
    # the anomalies during the heatwave period forecasted by each initiailzation
    # verification file is the same for all forecasts. Plots are saved to the 
    # indicated output directory
    for forecast in config.forecast_params: 

        # check if cache for climo exists, if not calculate it
        fcst_climatology_file = forecast.file.replace('.nc', '_ndvi-clima.nc')
        if not os.path.exists(fcst_climatology_file):

            logger.info(f"Calculating climatology for {forecast.file} and caching to {fcst_climatology_file}")
            # open the forecast file
            fcst = xr.open_dataset(forecast.file).NDVI_gapfill
            # resolve the step dimension into valid time
            valid_time = fcst.time.values + fcst.step.values
            fcst = fcst.assign_coords({'step': valid_time})
            fcst = fcst.squeeze().reset_coords(drop=True).rename({'step': 'time'}) # clean up coords/dims

            # calculate seasonal climatology from the forecast data
            fcst_climatology = fcst.groupby('time.dayofyear').mean(dim='time')

            # cache the climatology to a file
            fcst_climatology.to_netcdf(fcst_climatology_file)
        
        # load the forecast climatology
        logger.info(f"Loading cached climatology from {fcst_climatology_file}")
        fcst_climatology = xr.open_dataset(fcst_climatology_file).NDVI_gapfill  
        # average climo into seasons: DJF, MAM, JJA, SON from day of year values
        fcst_climatology = fcst_climatology.assign_coords(dayofyear = pd.date_range('2000-01-01', '2000-12-31', freq='D')).groupby('dayofyear.season').mean(dim='dayofyear')
        # plot seasons from forecast climatology
        forecast_climatology_file_prefix = config.output_directory + f'{forecast.model_id}_climatology'
        logger.info(f"Plotting forecast climatology to {forecast_climatology_file_prefix}*")
        _plot_global(fcst_climatology, mapper, forecast_climatology_file_prefix)

    return
    


if __name__ == "__main__":

    # receive arguments 
    import argparse
    parser = argparse.ArgumentParser(description="Plot climatology of NDVI over historical run.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # run analyses 
    main(args.config)