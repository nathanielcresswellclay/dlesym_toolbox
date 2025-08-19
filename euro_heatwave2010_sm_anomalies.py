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
import cartopy.feature as cfeature

# mounted modules 
from toolbox_utils import setup_logging

def main(config_path: str):

    """
    send config to routines that plot swvl1 anomalies during the 2010 heatwave in Europe

    """

    # load the config file
    config = OmegaConf.load(config_path)
    _, logger = setup_logging(config.output_log)

    logger.info("Loaded config:\n\n" + OmegaConf.to_yaml(config))  # prints the loaded YAML for clarity

    print(type(config))
    _plot_sm_anoms(config, logger)
    logger.info("Finished plotting soil moisture anomalies for the 2010 heatwave in Europe.")


# colormap with white as the middle color, coolwarm base
def _get_custom_cmap_brg():
    """
    Create a custom colormap with white as the first color.
    """
    # Get the 'coolwarm' colormap
    bwr = cm.get_cmap('BrBG')
    # Create a new colormap with white as the middle color
    new_colors = bwr(np.linspace(0, 1, 256))
    for i in range(118,139): new_colors[i] = mcolors.to_rgba('whitesmoke')  # RGBA for white
    new_cmap = mcolors.ListedColormap(new_colors)
    return new_cmap

def _plot_sm_anoms(config: DictConfig , logger: logging.Logger):
    
    # try to import remap module from DLESyM 
    # NOTE there are some brittle dependencies to get remap to work properly
    # for environment recipe: https://github.com/AtmosSci-DLESM/DLESyM/blob/main/environments/dlesym-0.1.yaml
    try: 
        import data_processing.remap.healpix as hpx 
    except ImportError as e:
        logger.error("Failed to import reamp data_processing.remap.healpix")
        logger.error(str(e))
        return  
    
    # load our verification data and select swvl1 during the heatwave
    verif_sm = xr.open_zarr(config.verification_file).targets.sel(
        time=pd.date_range('2010-07-25', '2010-08-08', freq='2D'),
        channel_out="swvl1"
    )

    # next we need to get the climatology for the same period. If we already have 
    # a climaology file cached, we can load it, otherwise we need to calculate it
    climatology_file = config.verification_file.replace('.zarr', '_swvl1-clima.nc')
    if os.path.exists(climatology_file):
        logger.info(f"Loading cached climatology from {climatology_file}")
        climatology = xr.open_dataset(climatology_file).targets
    else:
        logger.info("Calculating climatology for the heatwave period")
        climatology = xr.open_zarr(config.verification_file).targets.sel(
            channel_out="swvl1"
        ).groupby('time.dayofyear').mean(dim='time')
        climatology.to_netcdf(climatology_file)
        logger.info(f"Climatology saved to {climatology_file}")
    # select heatwave period from climatology. climatology time dim is dayofyear
    # so we need to select the days corresponding to the heatwave, then fix dims
    climatology = climatology.sel(
        dayofyear=pd.date_range('2010-07-25', '2010-08-08', freq='2D').dayofyear
    ).rename({"dayofyear": "time"}).assign_coords(
        time=pd.date_range('2010-07-25', '2010-08-08', freq='2D')
    )

    # now we calculate the observed anomalies, and average over time
    verif_sm_anomaly = verif_sm - climatology
    verif_sm_anomaly = verif_sm_anomaly.mean(dim='time').reset_coords(drop=True)
    # we also need to plot on lat lon grid, so we need to remap the data,
    # this remapper will be used for forecasts as well. Use 1 degree resolution
    mapper = hpx.HEALPixRemap(
        latitudes=181,
        longitudes=360,
        nside=64,
    )
    # remap the verification data to lat lon grid
    verif_sm_anomaly_ll = xr.DataArray(
        mapper.hpx2ll(verif_sm_anomaly.values),
        dims=['lat', 'lon'], 
        coords={'lat': np.arange(90, -90.1, -1),'lon': np.arange(0, 360, 1)}
    )

    # now we loop through to given forecasts and for each one calculate and plot
    # the anomalies during the heatwave period forecasted by each initiailzation
    # verification file is the same for all forecasts. Plots are saved to the 
    # indicated output directory
    for forecast in config.forecast_params: 

        # open the forecast file
        fcst = xr.open_dataset(forecast.file).swvl1.sel(time=config.inits)
        
        # intiiailize figure to hold plot. We need a panel for each init time +1 for the verif
        fig, axs = plt.subplots(
            ncols=1, nrows=len(config.inits) + 1, 
            figsize=(5, 4*(len(config.inits) + 1)), subplot_kw={'projection': ccrs.Robinson()}
        )
        for ax in axs: ax.set_extent([5, 80, 30, 72], crs=ccrs.PlateCarree()) ; ax.coastlines() # zoom in on europe and add coastlines

        # add cyclic point to the verification data for plotting
        verif_sm_anomaly_ll_cyclic, lon_cyclic = add_cyclic_point(
            verif_sm_anomaly_ll.values, coord=verif_sm_anomaly_ll.lon
        )

        # plot observed_anomaly
        im = axs[0].contourf(
            lon_cyclic, verif_sm_anomaly_ll.lat, 
            verif_sm_anomaly_ll_cyclic, 
            transform=ccrs.PlateCarree(), 
            cmap=_get_custom_cmap_brg(), 
            levels=np.arange(-.18,.19,.02),
            extend='both',
        )
        axs[0].set_title('ERA5 Anomalies: 2010-07-25 to 2010-08-08', fontsize=15)
        # Mask oceans by covering them with a white patch
        axs[0].add_feature(cfeature.OCEAN, zorder=10, facecolor='white')

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
            try:
                fcst_init = fcst_init.sel(time=pd.date_range('2010-07-25', '2010-08-08', freq='2D')).groupby('time.day').mean('time').rename({'day': 'time'}).assign_coords(
                    time=pd.date_range('2010-07-25', '2010-08-08', freq='2D')
                )
            except KeyError as e:
                logger.error(f"Error selecting forecast data for {init}. This can happen in chosen initialization time does not align with calculations of target anomalies. \
here we assume 2 day resolution inf the land model and forecast values for pd.date_range('2010-07-25', '2010-08-08', freq='2D'): {e}")
                continue

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
                cmap=_get_custom_cmap_brg(), 
                levels=np.arange(-.18,.19,.02),
                extend='both',
            )
            axs[i+1].set_title(f'Initialized: {init}', fontsize=15)
            
            # Mask oceans by covering them with a white patch
            axs[i+1].add_feature(cfeature.OCEAN, zorder=10, facecolor='white')

        # create axis for colorbar
        with warnings.catch_warnings(): 
            warnings.simplefilter("ignore")
            plt.tight_layout()
        cbar_ax = fig.add_axes([0.05, -0.02, 0.9, 0.015])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='Soil Moisture Anomaly  (m$^3$ m$^{-3}$)', ticks=np.arange(-.15,.16,.05))

        # save
        plot_file = config.output_directory + f'/swvl1_anomaly_{forecast.model_id}.png'
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