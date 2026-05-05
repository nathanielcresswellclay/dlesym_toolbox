import os
import sys
import time
import logging
import warnings
from tqdm import tqdm

# analysis
import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import OmegaConf

# plotting 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import cartopy.io.shapereader as shpreader

# mounted modules 
from toolbox_utils import setup_logging
from plotting_params_library import rmse_acc_scoring_variable_metas as var_metas

def open_verif_as_forecast(verif_file: str, forecast: xr.Dataset, logger: logging.Logger = None):
    """
    Open the verif and align with forecast for convienitent comparison
    """
    file_type = verif_file.split('.')[-1]
    if file_type == 'zarr':
        verif = xr.open_dataset(verif_file, engine='zarr')
    elif file_type == 'nc':
        verif = xr.open_dataset(verif_file, engine='netcdf')
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # find channels in forecast 
    channels = list(forecast.data_vars)
    verif = verif.targets.sel(channel_out=channels)

    # create buffer for verif forecast
    verif_forecast = xr.zeros_like(forecast)
    
    # populate verif forecast with verif data
    if logger is not None:
        logger.info(f"Populating verif forecast with verif data")
    
    for t in tqdm(verif_forecast.time, desc=" initializations"):
        for s in verif_forecast.step:

            time_stamp = t.values + s.values
            for channel in channels:
                verif_forecast[channel].loc[dict(time=t, step=s)] = verif.sel(time=time_stamp, channel_out=channel).values.squeeze()

    # return the verif data
    return verif_forecast

def calculate_acc(forecast: xr.Dataset, verif: xr.Dataset, climo: xr.Dataset, spatial_weights: xr.DataArray = None):
    """
    Calculate the ACC between the forecast and verif.
    Forecast, verif datasets have dimensions (time, step, face, height, width).
      - time is initialziation 
      - step is leadtime
      - face is the face of the healypix grid
      - height is the height of the healypix face
      - width is the width of the healypix face
    spatial_weights is a dataarray with dimensions (face, height, width) that is used to
    weight the ACC by the spatial dimensions.
    """
    # calculate anomalies 
    forec_anom = forecast - climo
    verif_anom = verif - climo
    
    # calculate acc
    if spatial_weights is not None:
        # create weights dataarray
        expanded_weights = xr.DataArray(
            np.stack((np.stack((spatial_weights,)*len(forecast.step), axis=0),)*len(forecast.time), axis=0), # we have to expand the weights here to match the forecast dims
            dims=['time', 'step', 'face', 'height', 'width'], 
            coords={'time': forecast.time, 'step': forecast.step, 'face': forecast.face, 'height': forecast.height, 'width': forecast.width})
        axis_mean = ['time', 'face', 'height','width']
        acc = ((verif_anom * forec_anom * expanded_weights).mean(dim=axis_mean, skipna=True)
            / np.sqrt((expanded_weights * verif_anom**2).mean(dim=axis_mean, skipna=True) *
                    (expanded_weights * forec_anom**2).mean(dim=axis_mean, skipna=True)))
    else:
        axis_mean = ['time', 'face', 'height','width']
        acc = ((verif_anom * forec_anom).mean(dim=axis_mean, skipna=True)
            / np.sqrt((verif_anom**2).mean(dim=axis_mean, skipna=True) *
                    (forec_anom**2).mean(dim=axis_mean, skipna=True)))
    
    return acc

def calculate_rmse(forecast: xr.Dataset, verif: xr.Dataset, spatial_weights: xr.DataArray = None):
    """
    Calculate the RMSE between the forecast and verif across spatial dimensions.
    Forecast, verif datasets have dimensions (time, step, face, height, width).
      - time is initialziation 
      - step is leadtime
      - face is the face of the healypix grid
      - height is the height of the healypix face
      - width is the width of the healypix face
    spatial_weights is a dataarray with dimensions (face, height, width) that is used to
    weight the RMSE by the spatial dimensions.
    """
    # calculate rmse 
    if spatial_weights is not None:
        # create weights dataarray
        expanded_weights = xr.DataArray(
            np.stack((np.stack((spatial_weights,)*len(forecast.step), axis=0),)*len(forecast.time), axis=0), # we have to expand the weights here to match the forecast dims
            dims=['time', 'step', 'face', 'height', 'width'], 
            coords={'time': forecast.time, 'step': forecast.step, 'face': forecast.face, 'height': forecast.height, 'width': forecast.width})
        # calculate rmse weighted by spatial weights
        rmse = np.sqrt( (((forecast - verif) ** 2) * expanded_weights).sum(dim=['time', 'face', 'height', 'width'], skipna=True) / expanded_weights.sum(dim=['time', 'face', 'height', 'width'], skipna=True) )
    else:
        rmse = np.sqrt(((forecast - verif) ** 2).mean(dim=['time', 'face', 'height','width'], skipna=True))
    
    return rmse

def main(config_path: str):

    """
    Run RMSE and ACC scoring for forecasts.

    ``forecasts`` must be a list of dictionaries with keys ``forecast_file``, ``verif_file``, ``label``, and ``plotting_kwargs``.
    """

    # load the config file
    config = OmegaConf.load(config_path)
    _, logger = setup_logging(config.output_log)

    logger.info("Loaded config:\n\n" + OmegaConf.to_yaml(config))  # prints the loaded YAML for clarity

    # initialize RMSE and ACC figures 
    rmse_fig, rmse_axs = plt.subplots(config.nrows, config.ncols, figsize=(config.ncols * 3.5, config.nrows * 3.5))
    acc_fig, acc_axs = plt.subplots(config.nrows, config.ncols, figsize=(config.ncols * 3.5, config.nrows * 3.5))

    # load and score forecasts
    for i, forecast_params in enumerate(config.forecasts):
        
        # check for cached ACC skill
        acc_skill_cache_path = forecast_params.skill_cache_prefix + "_acc.nc"
        if os.path.exists(acc_skill_cache_path):
            logger.info(f"Loading cached ACC skill from {acc_skill_cache_path}")
        else:
            logger.info(f"No cached ACC skill found, calculating and saving to {acc_skill_cache_path}")
            # load forecast_file, verif, and climo
            forecast = xr.open_dataset(forecast_params.forecast_file)[list(config.variables)]
            verif = xr.open_dataset(forecast_params.verif_file)
            climo = xr.open_dataset(forecast_params.climo_file)
            # calculate ACC
            acc_skill = calculate_acc(forecast, verif, climo, spatial_weights=getattr(config, 'spatial_weights', None))

            # cache ACC skill
            os.makedirs(os.path.dirname(acc_skill_cache_path), exist_ok=True)
            acc_skill.to_netcdf(acc_skill_cache_path)
            logger.info(f"Finished calculating ACC skill and caching ACC")
        acc = xr.open_dataset(acc_skill_cache_path)
    
        # check for cached RMSE skill
        rmse_skill_cache_path = forecast_params.skill_cache_prefix + "_rmse.nc"
        if os.path.exists(rmse_skill_cache_path):
            logger.info(f"Loading cached RMSE skill from {rmse_skill_cache_path}")
        else:
            logger.info(f"No cached RMSE skill found, calculating and saving to {rmse_skill_cache_path}")
            # load forecast_file and verif
            forecast = xr.open_dataset(forecast_params.forecast_file)[list(config.variables)]
            verif = xr.open_dataset(forecast_params.verif_file)

            # calculate RMSE
            rmse_skill = calculate_rmse(forecast, verif, spatial_weights=getattr(config, 'spatial_weights', None))

            # cache RMSE skill
            os.makedirs(os.path.dirname(rmse_skill_cache_path), exist_ok=True)
            rmse_skill.to_netcdf(rmse_skill_cache_path)
            logger.info(f"Finished calculating RMSE skill and caching RMSE")
        rmse = xr.open_dataset(rmse_skill_cache_path)
        
        # function for axis styling
        def style_axis(ax, var, skill_type: str, lib: dict):
            
            title = f"{var} ({lib['units']})" if skill_type == 'rmse' else f"{var}"
            ax.set_title(title)
            # ax.set_xlabel("Leadtime")
            # convert nano seconds time delta to days 
            leadtime_ticks = np.array(config.leadtime_ticks) * 24 * 60 * 60 * 1e9
            ax.set_xticks(leadtime_ticks)
            ax.set_xticklabels(config.leadtime_ticks)
            # set extent of x-axis to leadtime ticks
            ax.set_xlim(leadtime_ticks[0], leadtime_ticks[-1])

            # if rmse, round the y-axis labels to 2 decimal places
            if skill_type == 'rmse':
                if not np.all(np.mod(ax.get_yticks(), 1) == 0):
                    ax.set_yticklabels([f"{y:.2f}" for y in ax.get_yticks()])
            # gridlines for quantitative comparison
            ax.grid(color='black', linewidth=0.5)


            return ax

        # plot ACC and RMSE
        for v, var in enumerate(config.variables):
            lib = var_metas[var]
            acc_axs.flatten()[v].plot(acc.step, acc[var].values, **forecast_params.plotting_kwargs)
            style_axis(acc_axs.flatten()[v], var, 'acc', lib)
            rmse_axs.flatten()[v].plot(rmse.step, rmse[var].values * lib['scale_factor'], **forecast_params.plotting_kwargs)
            style_axis(rmse_axs.flatten()[v], var, 'rmse', lib)

        # draw legend on the last axis
        acc_axs.flatten()[-1].legend(loc='upper right')
        rmse_axs.flatten()[-1].legend(loc='lower right')

    # save plots 
    format = "png" if not getattr(config, 'vectorized', False) else "pdf"
    acc_filename = os.path.join(config.output_directory, config.output_file_prefix + "_acc" + "." + format)
    rmse_filename = os.path.join(config.output_directory, config.output_file_prefix + "_rmse" + "." + format)
    logger.info(f"Saving plots to {acc_filename} and {rmse_filename}")
    os.makedirs(config.output_directory, exist_ok=True)
    acc_fig.tight_layout()
    rmse_fig.tight_layout()
    acc_fig.savefig(acc_filename, dpi=300)
    rmse_fig.savefig(rmse_filename, dpi=300)

if __name__ == "__main__":

    # receive arguments 
    import argparse
    parser = argparse.ArgumentParser(description="Plot .")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # run analyses 
    main(args.config)

    