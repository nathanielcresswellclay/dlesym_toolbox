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
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import cartopy.io.shapereader as shpreader

# mounted modules 
from toolbox_utils import setup_logging

# Verification zarr `channel_out` and forecast NetCDF variable name
T2M_CHANNEL = "t2m"


def main(config_path: str):

    """
    Plot seasonal climatology of 2 m temperature (verification and forecasts).
    """

    # load the config file
    config = OmegaConf.load(config_path)
    _, logger = setup_logging(config.output_log)

    logger.info("Loaded config:\n\n" + OmegaConf.to_yaml(config))  # prints the loaded YAML for clarity

    _plot_t2m_climo(config, logger)
    logger.info("Finished plotting climatology for 2 m temperature.")


def _get_t2m_cmap():
    """Sequential colormap for absolute temperature."""
    return cm.get_cmap("RdYlBu_r")

# colormap with white as the middle color, bwr base
def _get_custom_cmap_bwr():
    """
    Create a custom colormap with white as the first color.
    """
    # Get the 'coolwarm' colormap
    bwr = cm.get_cmap('bwr_r')
    # Create a new colormap with white as the middle color
    new_colors = bwr(np.linspace(0, 1, 256))
    for i in range(112,147): new_colors[i] = mcolors.to_rgba('whitesmoke')  # RGBA for white
    new_cmap = mcolors.ListedColormap(new_colors)
    return new_cmap

def _plot_global(data: xr.DataArray, mapper, output_file_prefix: str, diff_plot: bool=False):
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
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax.coastlines(zorder=16)

        # add cyclic point to the data for plotting
        data_ll_cyclic, lon_cyclic = add_cyclic_point(
            data_ll, coord=data_ll.lon
        )
        
        # 2 m temperature in Kelvin (ERA5-style); adjust if your data are Celsius
        im = ax.contourf(
            lon_cyclic, data_ll.lat, data_ll_cyclic,
            transform=ccrs.PlateCarree(),
            cmap=_get_t2m_cmap() if not diff_plot else _get_custom_cmap_bwr(),
            levels=np.arange(220, 318, 4) if not diff_plot else np.arange(-8.0, 8.5, 0.5),
            extend='both',
            zorder=5,
        )

        # add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
        cbar.set_label(
            '2 m temperature (K)' if not diff_plot else '2 m temperature difference (K)',
            fontsize=12,
        )
        # title and save the figure
        plt.title(f"{season}", fontsize=15)
        plt.savefig(f"{output_file_prefix}_{season}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    return

def _plot_t2m_climo(config: DictConfig , logger: logging.Logger):
    
    # try to import remap module from DLESyM 
    # NOTE there are some brittle dependencies to get remap to work properly
    # for environment recipe: https://github.com/AtmosSci-DLESM/DLESyM/blob/main/environments/dlesym-0.1.yaml
    try: 
        import data_processing.remap.healpix as hpx 
    except ImportError as e:
        logger.error("Failed to import reamp data_processing.remap.healpix")
        logger.error(str(e))
        return  

    climatology_file = config.verification_file.replace('.zarr', f'_{T2M_CHANNEL}-clima.nc')
    if os.path.exists(climatology_file):
        logger.info(f"Loading cached climatology from {climatology_file}")
        climatology = xr.open_dataset(climatology_file).targets
    else:
        logger.info(f"Calculating climatology from {config.verification_file}")
        climatology = xr.open_zarr(config.verification_file).targets.sel(
            channel_out=T2M_CHANNEL
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

    # create output directory if it doesn't exist
    os.makedirs(config.output_directory, exist_ok=True)

    obs_climatology_file_prefix = config.output_directory + f'/obs_{T2M_CHANNEL}-climatology'
    logger.info(f"Plotting observed climatology to {obs_climatology_file_prefix}*")
    _plot_global(climatology, mapper, obs_climatology_file_prefix)

    for forecast in config.forecast_params: 

        fcst_climatology_file = forecast.file.replace('.nc', f'_{T2M_CHANNEL}-clima.nc')
        if not os.path.exists(fcst_climatology_file):

            logger.info(f"Calculating climatology for {forecast.file} and caching to {fcst_climatology_file}")
            fcst = xr.open_dataset(forecast.file)[T2M_CHANNEL]
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
        fcst_climatology = xr.open_dataset(fcst_climatology_file)[T2M_CHANNEL]
        # average climo into seasons: DJF, MAM, JJA, SON from day of year values
        fcst_climatology = fcst_climatology.assign_coords(dayofyear = pd.date_range('2000-01-01', '2000-12-31', freq='D')).groupby('dayofyear.season').mean(dim='dayofyear')
        forecast_climatology_file_prefix = config.output_directory + f'{forecast.model_id}_{T2M_CHANNEL}-climatology'
        logger.info(f"Plotting forecast climatology to {forecast_climatology_file_prefix}*")
        _plot_global(fcst_climatology, mapper, forecast_climatology_file_prefix)
        # plot difference map
        diff_climatology_file_prefix = config.output_directory + f'{forecast.model_id}_{T2M_CHANNEL}-climatology-diff'
        _plot_global(fcst_climatology - climatology, mapper, diff_climatology_file_prefix, diff_plot=True)


    return
    


if __name__ == "__main__":

    # receive arguments 
    import argparse
    parser = argparse.ArgumentParser(
        description="Plot seasonal climatology of 2 m surface temperature (t2m)."
    )
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # run analyses 
    main(args.config)
