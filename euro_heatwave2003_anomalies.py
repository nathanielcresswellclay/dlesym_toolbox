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
from cartopy.util import add_cyclic_point

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

    print(type(config))
    # _plot_t2m_anoms(config, logger)
    _plot_z500_anoms(config, logger)
    logger.info("Finished plotting anomalies for the 2003 heatwave in Europe.")


# colormap with white as the middle color, coolwarm base
def _get_custom_cmap_bwr():
    """
    Create a custom colormap with white as the first color.
    """
    # Get the 'coolwarm' colormap
    bwr = cm.get_cmap('coolwarm')
    # Create a new colormap with white as the middle color
    new_colors = bwr(np.linspace(0, 1, 256))
    for i in range(118,139): new_colors[i] = mcolors.to_rgba('whitesmoke')  # RGBA for white
    new_cmap = mcolors.ListedColormap(new_colors)
    return new_cmap

# colormap with white as the middle color, spectral_r base
def _get_custom_cmap_spectral():
    """
    Create a custom colormap with white as the first color.
    """
    # Get the 'spectral_r' colormap
    bwr = cm.get_cmap('Spectral_r')
    # Create a new colormap with white as the middle color
    new_colors = bwr(np.linspace(0, 1, 256))
    for i in range(118,139): new_colors[i] = mcolors.to_rgba('whitesmoke')  # RGBA for white
    new_cmap = mcolors.ListedColormap(new_colors)
    return new_cmap

def _plot_t2m_anoms(config: DictConfig , logger: logging.Logger):
    
    # try to import remap module from DLESyM 
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
        time=pd.date_range('2003-08-06', '2003-08-12', freq='6H'),
        channel_out="t2m0"
    )

    # we're calculating daily anomalies, so we need to average over the 6-hourly data
    # then reformat the output of groupby to have useful time coordinates
    verif_temp = verif_temp.groupby("time.day").mean("time").rename({"day": "time"}).assign_coords(
        time=pd.date_range('2003-08-06', '2003-08-12', freq='D')
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
        dayofyear=pd.date_range('2003-08-06', '2003-08-12', freq='D').dayofyear
    ).rename({"dayofyear": "time"}).assign_coords(
        time=pd.date_range('2003-08-06', '2003-08-12', freq='D')
    )

    # now we calculate the observed anomalies, and average over time
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
        coords={'lat': np.arange(90, -90.1, -1),'lon': np.arange(0, 360, 1)}
    )

    # now we loop through to given forecasts and for each one calculate and plot
    # the anomalies during the heatwave period forecasted by each initiailzation
    # verification file is the same for all forecasts. Plots are saved to the 
    # indicated output directory
    for forecast in config.forecast_params: 

        # open the forecast file
        fcst = xr.open_dataset(forecast.file).t2m0.sel(time=config.inits)
        
        # intiiailize figure to hold plot. We need a panel for each init time +1 for the verif
        fig, axs = plt.subplots(
            ncols=1, nrows=len(config.inits) + 1, 
            figsize=(5, 4*(len(config.inits) + 1)), subplot_kw={'projection': ccrs.Robinson()}
        )
        for ax in axs: ax.set_extent([-25, 35, 34, 72], crs=ccrs.PlateCarree()) ; ax.coastlines() # zoom in on europe and add coastlines

        # add cyclic point to the verification data for plotting
        verif_temp_anomaly_ll_cyclic, lon_cyclic = add_cyclic_point(
            verif_temp_anomaly_ll.values, coord=verif_temp_anomaly_ll.lon
        )

        # plot observed_anomaly
        im = axs[0].contourf(
            lon_cyclic, verif_temp_anomaly_ll.lat, 
            verif_temp_anomaly_ll_cyclic, 
            transform=ccrs.PlateCarree(), 
            cmap=_get_custom_cmap_bwr(), 
            levels=np.arange(-10,10.1,1),
            extend='both',
        )
        axs[0].set_title('ERA5 Anomalies: 2003-08-06 to 2003-08-12', fontsize=15)

        # now we loop through the initialization and plot the forecasted anomalies 
        # during the target week 
        for i, init in enumerate(config.inits):

            # select the forecast for the current initialization
            fcst_init = fcst.sel(time=init)
            # resolve the step dimension into valid time
            valid_time = fcst_init.time.values + fcst_init.step.values
            fcst_init = fcst_init.assign_coords({'step': valid_time})
            fcst_init = fcst_init.squeeze().reset_coords(drop=True).rename({'step': 'time'}) # clean up coords/dims

            # select target period and calculate daily mean, also format time time for compatibility with climatology
            fcst_init = fcst_init.sel(time=pd.date_range('2003-08-06', '2003-08-12', freq='D')).groupby('time.day').mean('time').rename({'day': 'time'}).assign_coords(
                time=pd.date_range('2003-08-06', '2003-08-12', freq='D')
            )

            # calculate the anomaly for the forecast
            fcst_anomaly = fcst_init - climatology
            fcst_anomaly = fcst_anomaly.mean(dim='time').reset_coords(drop=True)
            # remap the forecast anomaly to lat lon grid
            fcst_anomaly_ll = xr.DataArray(
                mapper.hpx2ll(fcst_anomaly.values),
                dims=['lat', 'lon'], 
                coords={'lat': np.arange(90, -90.1, -1),'lon': np.arange(0, 360, 1)}
            )
            # add cyclic point to the forecast data for plotting
            fcst_anomaly_ll_cyclic, lon_cyclic = add_cyclic_point(
                fcst_anomaly_ll.values, coord=fcst_anomaly_ll.lon
            )
            # plot the forecast anomaly
            axs[i+1].contourf(
                lon_cyclic, fcst_anomaly_ll.lat, 
                fcst_anomaly_ll_cyclic, 
                transform=ccrs.PlateCarree(), 
                cmap=_get_custom_cmap_bwr(), 
                levels=np.arange(-10,10.1,1),
                extend='both',
            )
            axs[i+1].set_title(f'Initialized: {init}', fontsize=15)

        # create axis for colorbar
        with warnings.catch_warnings(): 
            warnings.simplefilter("ignore")
            plt.tight_layout()
        cbar_ax = fig.add_axes([0.05, -0.02, 0.9, 0.015])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='2m Temperature Anomaly (K)')

        # save
        plot_file = config.output_directory + f'/t2m0_anomaly_{forecast.model_id}.png'
        fig.savefig(plot_file, bbox_inches='tight',dpi=300)
        logger.info(f"Saved anomaly plot to {plot_file}")
    
def _plot_z500_anoms(config: DictConfig, logger: logging.Logger):
    
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
    verif_z500 = xr.open_zarr(config.verification_file).targets.sel(
        time=pd.date_range('2003-08-06', '2003-08-12', freq='6H'),
        channel_out="z500"
    )

    # we're calculating daily anomalies, so we need to average over the 6-hourly data
    # then reformat the output of groupby to have useful time coordinates
    verif_z500 = verif_z500.groupby("time.day").mean("time").rename({"day": "time"}).assign_coords(
        time=pd.date_range('2003-08-06', '2003-08-12', freq='D')
    )
    # next we need to get the climatology for the same period. If we already have 
    # a climaology file cached, we can load it, otherwise we need to calculate it
    climatology_file = config.verification_file.replace('.zarr', '_z500-clima.nc')
    if os.path.exists(climatology_file):
        logger.info(f"Loading cached climatology from {climatology_file}")
        climatology = xr.open_dataset(climatology_file).targets
    else:
        logger.info("Calculating climatology for the heatwave period")
        climatology = xr.open_zarr(config.verification_file).targets.sel(
            channel_out="z500"
        ).groupby('time.dayofyear').mean(dim='time')
        climatology.to_netcdf(climatology_file)
        logger.info(f"Climatology saved to {climatology_file}")
    # select heatwave period from climatology. climatology time dim is dayofyear
    # so we need to select the days corresponding to the heatwave, then fix dims
    climatology = climatology.sel(
        dayofyear=pd.date_range('2003-08-06', '2003-08-12', freq='D').dayofyear
    ).rename({"dayofyear": "time"}).assign_coords(
        time=pd.date_range('2003-08-06', '2003-08-12', freq='D')
    )

    # now we calculate the observed anomalies, and average over time
    verif_z500_anomaly = verif_z500 - climatology
    verif_z500_anomaly = verif_z500_anomaly.mean(dim='time').reset_coords(drop=True)
    # we also need to plot on lat lon grid, so we need to remap the data,
    # this remapper will be used for forecasts as well. Use 1 degree resolution
    mapper = hpx.HEALPixRemap(
        latitudes=181,
        longitudes=360,
        nside=64,
    )
    # remap the verification data to lat lon grid
    verif_z500_anomaly_ll = xr.DataArray(
        mapper.hpx2ll(verif_z500_anomaly.values),
        dims=['lat', 'lon'], 
        coords={'lat': np.arange(90, -90.1, -1),'lon': np.arange(0, 360, 1)}
    )

    # now we loop through to given forecasts and for each one calculate and plot
    # the anomalies during the heatwave period forecasted by each initiailzation
    # verification file is the same for all forecasts. Plots are saved to the 
    # indicated output directory
    for forecast in config.forecast_params: 

        # open the forecast file
        fcst = xr.open_dataset(forecast.file).z500.sel(time=config.inits)
        
        # intiiailize figure to hold plot. We need a panel for each init time +1 for the verif
        fig, axs = plt.subplots(
            ncols=1, nrows=len(config.inits) + 1, 
            figsize=(5, 4*(len(config.inits) + 1)), subplot_kw={'projection': ccrs.Robinson()}
        )
        for ax in axs: ax.set_extent([-25, 35, 34, 72], crs=ccrs.PlateCarree()) ; ax.coastlines() # zoom in on europe and add coastlines

        # add cyclic point to the verification data for plotting
        verif_z500_anomaly_ll_cyclic, lon_cyclic = add_cyclic_point(
            verif_z500_anomaly_ll.values, coord=verif_z500_anomaly_ll.lon
        )

        # plot observed_anomaly
        im = axs[0].contourf(
            lon_cyclic, verif_z500_anomaly_ll.lat, 
            verif_z500_anomaly_ll_cyclic / 9.81,  # convert to geopotential height anomaly in m
            transform=ccrs.PlateCarree(), 
            cmap=_get_custom_cmap_spectral(), 
            levels=np.arange(-150,150.1,15),
            extend='both',
        )
        axs[0].set_title('ERA5 Anomalies: 2003-08-06 to 2003-08-12', fontsize=15)

        # now we loop through the initialization and plot the forecasted anomalies 
        # during the target week 
        for i, init in enumerate(config.inits):

            # select the forecast for the current initialization
            fcst_init = fcst.sel(time=init)
            # resolve the step dimension into valid time
            valid_time = fcst_init.time.values + fcst_init.step.values
            fcst_init = fcst_init.assign_coords({'step': valid_time})
            fcst_init = fcst_init.squeeze().reset_coords(drop=True).rename({'step': 'time'}) # clean up coords/dims

            # select target period and calculate daily mean, also format time time for compatibility with climatology
            fcst_init = fcst_init.sel(time=pd.date_range('2003-08-06', '2003-08-12', freq='D')).groupby('time.day').mean('time').rename({'day': 'time'}).assign_coords(
                time=pd.date_range('2003-08-06', '2003-08-12', freq='D')
            )

            # calculate the anomaly for the forecast
            fcst_anomaly = fcst_init - climatology
            fcst_anomaly = fcst_anomaly.mean(dim='time').reset_coords(drop=True)
            # remap the forecast anomaly to lat lon grid
            fcst_anomaly_ll = xr.DataArray(
                mapper.hpx2ll(fcst_anomaly.values),
                dims=['lat', 'lon'], 
                coords={'lat': np.arange(90, -90.1, -1),'lon': np.arange(0, 360, 1)}
            )
            # add cyclic point to the forecast data for plotting
            fcst_anomaly_ll_cyclic, lon_cyclic = add_cyclic_point(
                fcst_anomaly_ll.values, coord=fcst_anomaly_ll.lon
            )
            # plot the forecast anomaly
            axs[i+1].contourf(
                lon_cyclic, fcst_anomaly_ll.lat, 
                fcst_anomaly_ll_cyclic / 9.81,  # convert to geopotential height anomaly in m
                transform=ccrs.PlateCarree(), 
                cmap=_get_custom_cmap_spectral(), 
                levels=np.arange(-150,150.1,15),
                extend='both',
            )
            axs[i+1].set_title(f'Initialized: {init}', fontsize=15)

        # create axis for colorbar
        with warnings.catch_warnings(): 
            warnings.simplefilter("ignore")
            plt.tight_layout()
        cbar_ax = fig.add_axes([0.05, -0.02, 0.9, 0.015])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='Z$_{500}$ Anomaly (m)')

        # save
        plot_file = config.output_directory + f'/z500_anomaly_{forecast.model_id}.png'
        fig.savefig(plot_file, bbox_inches='tight',dpi=300)
        logger.info(f"Saved anomaly plot to {plot_file}")


if __name__ == "__main__":

    # receive arguments 
    import argparse
    parser = argparse.ArgumentParser(description="Run skill score comparison between forecasts.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # run analyses 
    main(args.config)