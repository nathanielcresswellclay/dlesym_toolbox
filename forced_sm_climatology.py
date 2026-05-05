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
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import cartopy.io.shapereader as shpreader
import matplotlib.ticker as mticker

# mounted modules 
from toolbox_utils import setup_logging
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def main(config_path: str):

    """
    send config to routines that plot t2m and z500 anomalies during the 2003 heatwave in Europe

    """

    # load the config file
    config = OmegaConf.load(config_path)
    _, logger = setup_logging(config.output_log)

    logger.info("Loaded config:\n\n" + OmegaConf.to_yaml(config))  # prints the loaded YAML for clarity

    _plot_sm_climo(config, logger)
    logger.info("Finished plotting climatology for Soil Moisture.")


# colormap with white as the middle color, coolwarm base
def _get_custom_cmap_blues():
    """
    Create a custom colormap with white as the first color.
    """
    # Get the 'coolwarm' colormap
    bwr = cm.get_cmap('Blues')
    # Create a new colormap with white as the middle color
    new_colors = bwr(np.linspace(0, 1, 256))
    for i in range(0,30): new_colors[i] = mcolors.to_rgba('whitesmoke')  # RGBA for white
    new_cmap = mcolors.ListedColormap(new_colors)
    return new_cmap

# colormap with white as the middle color, bwr base
def get_custom_cmap_bwr():
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

# helper function to plot the contourf and physical features
def land_contourf(lat, lon, data, levels, cmap):

    fig, ax = plt.subplots(
        figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()}
    )
    ax.add_feature(cfeature.OCEAN, zorder=10, facecolor="lightgrey")
    ax.add_feature(
        cfeature.LAKES,
        zorder=10,
        facecolor="lightgrey",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_extent([-180, 180, -70, 80], crs=ccrs.PlateCarree())
    ax.coastlines()
    
    im = ax.contourf(
        lon,
        lat,
        data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        levels=levels,
        extend="both",
    )
    shpfilename = shpreader.natural_earth(
        resolution="110m", category="cultural", name="admin_0_countries"
    )
    reader = shpreader.Reader(shpfilename)
    countries = reader.records()

    # Reduce spurious land/ice-sheet extent in coarse masks
    for country in countries:
        if country.attributes["NAME"] in ["Greenland", "Antarctica"]:
            ax.add_geometries(
                [country.geometry],
                crs=ccrs.PlateCarree(),
                facecolor="white",
                edgecolor="black",
                linewidth=0.5,
                zorder=11,
            )

    # Add gridlines and labels
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), 
        draw_labels=True,
        linewidth=.75, 
        color='black', 
        alpha=0.5, 
        linestyle='--',
        zorder=12,
    )

    # GRIDLINES
    grid_lon = [-180, -90, 0, 90, 180]
    grid_lat = [-45, 0, 45]

    # Specify exactly where you want the lines
    gl.xlocator = mticker.FixedLocator(grid_lon)
    gl.ylocator = mticker.FixedLocator(grid_lat)

    # Control label visibility (optional: hide top/right to look cleaner)
    gl.top_labels = False
    gl.right_labels = False

    # Standardize label formatting
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # Force the gridlines to the top of the stack
    gl.zorder = 20

    # Combined list for the side-ticks (10s and 5s)
    # We use these as "Minor" ticks so they don't overwrite the "Major" grid ticks
    sub_lon = np.arange(-180, 181, 5)
    sub_lat = np.arange(-70, 81, 5)

    # 2. SET THE XL TICKS (Major)
    # These align with your gridlines and the gl labels
    ax.set_xticks(grid_lon, crs=ccrs.PlateCarree())
    ax.set_yticks(grid_lat, crs=ccrs.PlateCarree())
    ax.tick_params(axis='both', which='major', length=6, width=1.0, zorder=25)

    # 3. SET THE 10s and 5s (Minor)
    ax.xaxis.set_minor_locator(mticker.FixedLocator(sub_lon))
    ax.yaxis.set_minor_locator(mticker.FixedLocator(sub_lat))
    ax.tick_params(axis='both', which='minor', length=3.5, width=0.5, zorder=25)

    # 4. FIX GL labels
    # This directly tells the label artists to move further away
    gl.xpadding = 10  # Horizontal padding for longitude
    gl.ypadding = 10  # Vertical padding for latitude
    
    # If the above still fails, force it via the label style dictionary:
    gl.xlabel_style = {'size': 10, 'color': 'black', 'va': 'top'}
    gl.ylabel_style = {'size': 10, 'color': 'black', 'ha': 'right'}

    # 5. Final Cleanup
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return fig, ax, im

def _plot_global(data: xr.DataArray, mapper, output_file_prefix: str, diff_plot: bool=False, vectorize: bool=False):
    """
    Plot global data on a map using the specified mapper.
    """

    if diff_plot:
        levels = np.arange(-0.2, 0.201, 0.02)
        cbar_ticks = np.arange(-0.2, 0.201, 0.1)
        cmap = get_custom_cmap_bwr()
        cbar_label = "Soil moisture bias (m$^3$m$^{-3}$)"

    else:
        levels = np.arange(0, .801, 0.05)
        cbar_ticks = np.arange(0, .801, 0.1)
        cmap = _get_custom_cmap_blues()
        cbar_label = "Soil moisture (m$^3$m$^{-3}$)"

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
        
        # plot the data
        fig, ax, im = land_contourf(
            lat=data_ll.lat, lon=lon_cyclic, data=data_ll_cyclic, 
            levels=levels, cmap=cmap)

        # format figure
        ax.set_title(f"{season} Climatology", fontsize=15)
        # add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, shrink=0.6, aspect=30)

        # format cbar 
        cbar.set_ticks(cbar_ticks)
        cbar.set_label(cbar_label, fontsize=12)

        # title and save the figure
        if vectorize:
            fig.savefig(f"{output_file_prefix}_{season}.pdf", dpi=300, bbox_inches='tight')
        else:
            fig.savefig(f"{output_file_prefix}_{season}.png", dpi=300, bbox_inches='tight')

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
    annual_fig, annual_ax, annual_im = land_contourf(
        lat=annual_data_ll.lat, lon=lon_cyclic, data=annual_data_ll_cyclic, 
        levels=levels, cmap=cmap)
    # colorbar
    cbar = plt.colorbar(annual_im, ax=annual_ax, orientation='horizontal', pad=0.1, shrink=0.6, aspect=30)
    cbar.set_ticks(cbar_ticks)
    cbar.set_label(cbar_label, fontsize=12)
    if vectorize:
        annual_fig.savefig(f"{output_file_prefix}_annual.pdf", dpi=300, bbox_inches='tight')
    else:
        annual_fig.savefig(f"{output_file_prefix}_annual.png", dpi=300, bbox_inches='tight')
    plt.close(annual_fig)
    return

    return

def _plot_sm_climo(config: DictConfig , logger: logging.Logger):
    
    # try to import remap module from DLESyM 
    # NOTE there are some brittle dependencies to get remap to work properly
    # for environment recipe: https://github.com/AtmosSci-DLESM/DLESyM/blob/main/environments/dlesym-0.1.yaml
    try: 
        import data_processing.remap.healpix as hpx 
    except ImportError as e:
        logger.error("Failed to import reamp data_processing.remap.healpix")
        logger.error(str(e))
        return  

    # create output directory if it doesn't exist
    os.makedirs(config.output_directory, exist_ok=True)

    # next we need to get the climatology of verif data for comparison
    climatology_file = config.verification_file.replace('.zarr', '_swvl1-clima.nc')
    if os.path.exists(climatology_file):
        logger.info(f"Loading cached climatology from {climatology_file}")
        climatology = xr.open_dataset(climatology_file).targets
    else:
        logger.info(f"Calculating climatology from {config.verification_file}")
        climatology = xr.open_zarr(config.verification_file).targets.sel(
            channel_out="swvl1"
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
    
    # plot seasons from observed climatology
    obs_climatology_file_prefix = config.output_directory + '/obs_sm-climatology'
    logger.info(f"Plotting observed climatology to {obs_climatology_file_prefix}*")
    _plot_global(climatology, mapper, obs_climatology_file_prefix)

    # now we loop through to given forecasts and for each one calculate and plot
    # the anomalies during the heatwave period forecasted by each initiailzation
    # verification file is the same for all forecasts. Plots are saved to the 
    # indicated output directory
    for forecast in config.forecast_params: 

        # check if cache for climo exists, if not calculate it
        fcst_climatology_file = forecast.file.replace('.nc', '_swvl1-clima.nc')
        if not os.path.exists(fcst_climatology_file):

            logger.info(f"Calculating climatology for {forecast.file} and caching to {fcst_climatology_file}")
            # open the forecast file
            fcst = xr.open_dataset(forecast.file).swvl1
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
        fcst_climatology = xr.open_dataset(fcst_climatology_file).swvl1  
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
        # plot seasons from forecast climatology
        forecast_climatology_file_prefix = config.output_directory + f'{forecast.model_id}_sm-climatology'
        logger.info(f"Plotting forecast climatology to {forecast_climatology_file_prefix}*")
        _plot_global(fcst_climatology, mapper, forecast_climatology_file_prefix)
        # plot difference map
        diff_climatology_file_prefix = config.output_directory + f'{forecast.model_id}_sm-climatology-diff'
        _plot_global(fcst_climatology - climatology, mapper, diff_climatology_file_prefix, diff_plot=True,
            vectorize=getattr(config, 'vectorize', False))


    return
    


if __name__ == "__main__":

    # receive arguments 
    import argparse
    parser = argparse.ArgumentParser(description="Plot climatology of soil moisture.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # run analyses 
    main(args.config)