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
from z500_climatology import contourf_atmos

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

def _plot_global(data: xr.DataArray, mapper, output_file_prefix: str, diff_plot: bool=False, vectorize: bool=False):
    """
    Plot global data on a map using the specified mapper.
    """

    if diff_plot:
        levels = np.arange(-10, 10.01, 1)
        cbar_ticks = np.arange(-10, 10.01, 5)
        cmap = _get_custom_cmap_bwr()
        cbar_label = "T$_{2m}$ difference (K)"
    else:
        levels = np.arange(230, 315.01, 5)
        cbar_ticks = np.arange(235, 310.01, 15)
        cmap = _get_t2m_cmap()
        cbar_label = "T$_{2m}$ (K)"

    for season in data.season.values:
        # select the data for the current season
        season_data = data.sel(season=season)
        # remap the data to lat/lon grid
        data_ll = xr.DataArray(
            mapper.hpx2ll(season_data.values),
            dims=['lat', 'lon'], 
            coords={'lat': np.arange(90, -90.1, -1),'lon': np.arange(0, 360, 1)}
        )

        # add cyclic point to the data for plotting
        data_ll_cyclic, lon_cyclic = add_cyclic_point(
            data_ll, coord=data_ll.lon
        )

        fig, ax, im = contourf_atmos(
            lat=data_ll.lat, lon=lon_cyclic, data=data_ll_cyclic, 
            levels=levels, cmap=cmap)

        ax.set_title(f"{season} Climatology", fontsize=15)

        # add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, shrink=0.6, aspect=30)
        cbar.set_ticks(cbar_ticks)
        cbar.set_label(cbar_label, fontsize=12)
        # title and save the figure
        if vectorize:
            plt.savefig(f"{output_file_prefix}_{season}.pdf", dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{output_file_prefix}_{season}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # ANNUAL PLOT 
    # note that generally avering is not associative but 
    #     since seasons have similar cardinality, 
    #     this approximation is valid.
    annual_data = data.mean(dim='season') 
    annual_data_ll = xr.DataArray(
        mapper.hpx2ll(annual_data.values),
        dims=['lat', 'lon'], 
        coords={'lat': np.arange(90, -90.1, -1),'lon': np.arange(0, 360, 1)}
    )
    annual_data_ll_cyclic, lon_cyclic = add_cyclic_point(
        annual_data_ll, coord=annual_data_ll.lon)
    annual_fig, annual_ax, annual_im = contourf_atmos(
        lat=annual_data_ll.lat, lon=lon_cyclic, data=annual_data_ll_cyclic, 
        levels=levels, cmap=cmap)
    cbar = plt.colorbar(annual_im, ax=annual_ax, orientation='horizontal', pad=0.1, shrink=0.6, aspect=30)
    cbar.set_ticks(cbar_ticks)
    cbar.set_label(cbar_label, fontsize=12)
    if vectorize:
        annual_fig.savefig(f"{output_file_prefix}_annual.pdf", dpi=300, bbox_inches='tight')
    else:
        annual_fig.savefig(f"{output_file_prefix}_annual.png", dpi=300, bbox_inches='tight')
    plt.close(annual_fig)
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

    climatology_file = config.verification_file.replace('.zarr', f'_t2m-clima.nc')
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

    obs_climatology_file_prefix = config.output_directory + f'/obs_t2m-climatology'
    logger.info(f"Plotting observed climatology to {obs_climatology_file_prefix}*")
    _plot_global(climatology, mapper, obs_climatology_file_prefix)

    for forecast in config.forecast_params: 

        fcst_climatology_file = forecast.file.replace('.nc', f'_t2m-clima.nc')
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
        # assign reference datetime values from 2000 for climo cycle
        ref_datetime = pd.date_range('2000-01-01', '2000-12-31', freq='D')
        # find day of year in both fcst_climatology and ref_datetime
        fcst_dayofyear = fcst_climatology.dayofyear.values
        ref_dayofyear = ref_datetime.dayofyear.values
        # find the index of the day of year in ref_datetime
        index = np.where(np.isin(ref_dayofyear, fcst_dayofyear))[0]
        # assign the day of year to the fcst_climatology
        fcst_climatology = fcst_climatology.assign_coords(dayofyear = ref_datetime[index])
        # group by season and mean over day of year
        fcst_climatology = fcst_climatology.groupby('dayofyear.season').mean(dim='dayofyear')
        forecast_climatology_file_prefix = config.output_directory + f'{forecast.model_id}_t2m-climatology'
        logger.info(f"Plotting forecast climatology to {forecast_climatology_file_prefix}*")
        _plot_global(fcst_climatology, mapper, forecast_climatology_file_prefix)
        # plot difference map
        diff_climatology_file_prefix = config.output_directory + f'{forecast.model_id}_t2m-climatology-diff'
        _plot_global(fcst_climatology - climatology, mapper, diff_climatology_file_prefix, diff_plot=True,
            vectorize=getattr(config, 'vectorize', False))


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
