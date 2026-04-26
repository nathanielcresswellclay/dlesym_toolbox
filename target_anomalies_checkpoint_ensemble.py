import os
import sys
import time
import glob
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
import data_processing.remap.healpix as hpx 
from plotting_params_library import plotting_param_library

# mounted modules 
from toolbox_utils import setup_logging

def main(config_path: str):

    """
    send config to routines that plot swvl1 anomalies during the 2003 heatwave in Europe

    """

    # load the config file
    config = OmegaConf.load(config_path)
    _, logger = setup_logging(config.output_log)

    logger.info("Loaded config:\n\n" + OmegaConf.to_yaml(config))  # prints the loaded YAML for clarity

    _plot_anomalies(config, logger)
    logger.info("Finished plotting anomalies during the target time slice for the checkpoint ensemble.")


# Routine for resolving and plotting anomalies for configurable sets of varialbes 
def _plot_anomalies(config: DictConfig , logger: logging.Logger):
    
    ########################################################################################
    # Internal plotting functions and objects
    ########################################################################################

    # for use in all plotting calls
    mapper = hpx.HEALPixRemap(
        latitudes=181,
        longitudes=360,
        nside=64,
    )
    # basis for all plotting calls in this function
    def _plot(anom: xr.DataArray, output_dir: str, output_file: str, var_name: str):

        # get plotting parameters for this variable
        lib = plotting_param_library[var_name]

        # remap the anomaly to lat lon grid
        anom_ll = xr.DataArray(
            mapper.hpx2ll(anom.values),
            dims=['lat', 'lon'], 
            coords={'lat': np.arange(90, -90.1, -1),'lon': np.arange(0, 360, 1)}
        )
        # add cyclic point to the anomaly
        # create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        # add cyclic point to the anomaly
        anom_cyclic, lon_cyclic = add_cyclic_point(anom_ll.values, coord=anom_ll.lon)

        # intiailize figure to hold plot. We need a panel for each init time +1 for the verif
        fig, ax = plt.subplots(
            ncols=1, nrows=1, subplot_kw={'projection': ccrs.Robinson()}
        )
        # extent of plot from config lat and lon ranges
        ax.set_extent([config.region.lat_range[0], config.region.lat_range[1], 
                       config.region.lon_range[0], config.region.lon_range[1]], 
                       crs=ccrs.PlateCarree())
        # add coastlines
        ax.coastlines() # zoom in on europe and add coastlines
        # mask oceans by covering them with a white patch
        ax.add_feature(cfeature.OCEAN, zorder=10, facecolor='lightgrey')

        # scale anomalies if scaling factor is in library
        if 'scaling' in lib:
            anom_cyclic = anom_cyclic * lib['scaling']

        # plot the anomaly
        im = ax.contourf(lon_cyclic, anom_ll.lat, anom_cyclic, 
                    transform=ccrs.PlateCarree(),
                    cmap=lib['cmap'], levels=lib['anomaly_levels'], extend='both')
        # add colorbar
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal',
                            label=(f'{lib["label"]} Anomaly ({lib["units"]})'),
                            ticks=lib['anomaly_ticks'],
                            shrink=0.7,
                            pad=0.05,
                            aspect=30)
        # save plot
        fig.savefig(os.path.join(output_dir, output_file), bbox_inches='tight',dpi=300)
        logger.info(f"Saved anomaly plot to {os.path.join(output_dir, output_file)}")
        plt.close()
    
    ########################################################################################
    # Land variables
    ########################################################################################

    # if land variables are prveded we calculate their anomalies during the target time slice
    if config.land_variables:
        logger.info("Calculating land variables anomalies for {} variables".format(len(config.land_variables)))
        # get list of forecast files in the ensemble 
        land_ensemble_files = glob.glob(os.path.join(config.ensemble_dir, 'land*.nc'))
        logger.info(f"Found {len(land_ensemble_files)} land forecast files in {config.ensemble_dir}")

        # resolve the time dimension for the target time slice by using forecast file time dimension
        land_forecast_temp = xr.open_dataset(land_ensemble_files[0])
        target_times = land_forecast_temp.time + land_forecast_temp.step
        target_times = target_times.assign_coords(step=target_times.values.squeeze())
        target_times = target_times.sel(step=slice(config.target_time_slice.start_time, config.target_time_slice.end_time)).step.values

        for variable in config.land_variables:

            # resolve climatology file fore this var 
            climatology_file = config.land_verification_file.replace('.zarr', f'_{variable}-clima.nc')
            if os.path.exists(climatology_file):
                logger.info(f"Loading cached climatology from {climatology_file}")
            else:
                logger.info(f"Calculating climatology for {variable} during the target time slice")
                climatology = xr.open_zarr(config.land_verification_file).targets.sel(
                    channel_out=variable).groupby('time.dayofyear').mean(dim='time')
                climatology.to_netcdf(climatology_file)
                logger.info(f"Climatology saved to {climatology_file}")
            # load climatology
            climatology = xr.open_dataset(climatology_file).targets
            # select target period from climatology. climatology time dim is dayofyear
            climatology = climatology.sel(dayofyear=[pd.to_datetime(t).dayofyear for t in target_times])

            # for each forecast file, calculate the anomaly
            ensemble_anomalies = []
            for i, forecast_file in enumerate(land_ensemble_files):

                # open file, select var
                forecast = xr.open_dataset(forecast_file)[variable]
                # assert initialization time dimension is only 1. More inits are not supported yet.
                assert len(forecast.time) == 1, "More than 1 initialization time is not supported yet"

                # select target times
                valid_time = forecast.time.values + forecast.step.values
                forecast = forecast.assign_coords({'step': valid_time})
                target_forecast = forecast.sel(step=target_times)
                # collapse singleton time dimension
                target_forecast = target_forecast.isel(time=0).squeeze()
                target_forecast_anomaly = target_forecast - climatology

                # average over time and append to ensemble anomalies
                target_forecast_anomaly = target_forecast_anomaly.mean(dim=('step', 'dayofyear')).squeeze()
                ensemble_anomalies.append(target_forecast_anomaly)

                # plot individual membes 
                if config.plot_individual_members:
                    _plot(
                        anom = target_forecast_anomaly,
                        output_dir = config.output_directory,
                        output_file = f'{variable}-anom_{i:02d}.png',
                        var_name = variable,
                    )
            
            # plot ensemble mean
            ensemble_mean_anomalies = xr.concat(ensemble_anomalies, dim='ensemble').mean(dim='ensemble')
            _plot(
                anom = ensemble_mean_anomalies,
                output_dir = config.output_directory,
                output_file = f'{variable}-anom_ensemble_mean.png',
                var_name = variable,
            )

            # plot observed anomalies
            verif_dataset = xr.open_zarr(config.land_verification_file).targets.sel(channel_out=variable)
            observed = verif_dataset.sel(time=target_times)
            observed = observed.assign_coords(time=pd.to_datetime(observed.time).dayofyear).rename({"time": "dayofyear"})
            observed_anomalies = observed - climatology
            # plot these 
            _plot(
                anom = observed_anomalies.mean(dim='dayofyear'),
                output_dir = config.output_directory,
                output_file = f'{variable}-anom_observed.png',
                var_name = variable,
            )

    ########################################################################################
    # Ocean variables
    ########################################################################################

    # if ocean variables are provided we calculate their anomalies during the target time slice
    if config.ocean_variables:
        logger.info("Calculating ocean variables anomalies for {} variables".format(len(config.ocean_variables)))
        # get list of forecast files in the ensemble 
        ocean_ensemble_files = glob.glob(os.path.join(config.ensemble_dir, 'ocean*.nc'))
        logger.info(f"Found {len(ocean_ensemble_files)} ocean forecast files in {config.ensemble_dir}")

        # resolve the time dimension for the target time slice by using forecast file time dimension
        ocean_forecast_temp = xr.open_dataset(ocean_ensemble_files[0])
        target_times = ocean_forecast_temp.time + ocean_forecast_temp.step
        target_times = target_times.assign_coords(step=target_times.values.squeeze())
        target_times = target_times.sel(step=slice(config.target_time_slice.start_time, config.target_time_slice.end_time)).step.values

        for variable in config.ocean_variables:

            # resolve climatology file fore this var 
            climatology_file = config.ocean_verification_file.replace('.zarr', f'_{variable}-clima.nc')
            if os.path.exists(climatology_file):
                logger.info(f"Loading cached climatology from {climatology_file}")
            else:
                logger.info(f"Calculating climatology for {variable} during the target time slice")
                climatology = xr.open_zarr(config.ocean_verification_file).targets.sel(
                    channel_out=variable).groupby('time.dayofyear').mean(dim='time')
                climatology.to_netcdf(climatology_file)
                logger.info(f"Climatology saved to {climatology_file}")
            # load climatology
            climatology = xr.open_dataset(climatology_file).targets
            # select target period from climatology. climatology time dim is dayofyear
            climatology = climatology.sel(dayofyear=[pd.to_datetime(t).dayofyear for t in target_times])

            # for each forecast file, calculate the anomaly
            ensemble_anomalies = []
            for i, forecast_file in enumerate(ocean_ensemble_files):

                # open file, select var
                forecast = xr.open_dataset(forecast_file)[variable]
                # assert initialization time dimension is only 1. More inits are not supported yet.
                assert len(forecast.time) == 1, "More than 1 initialization time is not supported yet"

                # select target times
                valid_time = forecast.time.values + forecast.step.values
                forecast = forecast.assign_coords({'step': valid_time})
                target_forecast = forecast.sel(step=target_times)
                # collapse singleton time dimension
                target_forecast = target_forecast.isel(time=0).squeeze()
                target_forecast_anomaly = target_forecast - climatology

                # average over time and append to ensemble anomalies
                target_forecast_anomaly = target_forecast_anomaly.mean(dim=('step', 'dayofyear')).squeeze()
                ensemble_anomalies.append(target_forecast_anomaly)

                # plot individual membes 
                if config.plot_individual_members:
                    _plot(
                        anom = target_forecast_anomaly,
                        output_dir = config.output_directory,
                        output_file = f'{variable}-anom_{i:02d}.png',
                        var_name = variable,
                    )
            
            # plot ensemble mean
            ensemble_mean_anomalies = xr.concat(ensemble_anomalies, dim='ensemble').mean(dim='ensemble')
            _plot(
                anom = ensemble_mean_anomalies,
                output_dir = config.output_directory,
                output_file = f'{variable}-anom_ensemble_mean.png',
                var_name = variable,
            )

            # plot observed anomalies
            verif_dataset = xr.open_zarr(config.ocean_verification_file).targets.sel(channel_out=variable)
            observed = verif_dataset.sel(time=target_times)
            observed = observed.assign_coords(time=pd.to_datetime(observed.time).dayofyear).rename({"time": "dayofyear"})
            observed_anomalies = observed - climatology
            # plot these 
            _plot(
                anom = observed_anomalies.mean(dim='dayofyear'),
                output_dir = config.output_directory,
                output_file = f'{variable}-anom_observed.png',
                var_name = variable,
            )
    
    ########################################################################################
    # Atmosphere variables
    ########################################################################################

    # if atmosphere variables are provided we calculate their anomalies during the target time slice
    if config.atmos_variables:
        logger.info("Calculating atmosphere variables anomalies for {} variables".format(len(config.atmos_variables)))
        # get list of forecast files in the ensemble 
        atmos_ensemble_files = glob.glob(os.path.join(config.ensemble_dir, 'atmos*.nc'))
        logger.info(f"Found {len(atmos_ensemble_files)} atmosphere forecast files in {config.ensemble_dir}")

        # resolve the time dimension for the target time slice by using forecast file time dimension
        atmos_forecast_temp = xr.open_dataset(atmos_ensemble_files[0])
        target_times = atmos_forecast_temp.time + atmos_forecast_temp.step
        target_times = target_times.assign_coords(step=target_times.values.squeeze())
        target_times = target_times.sel(step=slice(config.target_time_slice.start_time, config.target_time_slice.end_time)).step.values

        for variable in config.atmos_variables:

            # resolve climatology file fore this var 
            climatology_file = config.atmos_verification_file.replace('.zarr', f'_{variable}-clima.nc')
            if os.path.exists(climatology_file):
                logger.info(f"Loading cached climatology from {climatology_file}")
            else:
                logger.info(f"Calculating climatology for {variable} during the target time slice")
                climatology = xr.open_zarr(config.atmos_verification_file).targets.sel(
                    channel_out=variable).groupby('time.dayofyear').mean(dim='time')
                climatology.to_netcdf(climatology_file)
                logger.info(f"Climatology saved to {climatology_file}")
            # load climatology
            climatology = xr.open_dataset(climatology_file).targets
            # select target period from climatology. climatology time dim is dayofyear
            climatology = climatology.sel(dayofyear=[pd.to_datetime(t).dayofyear for t in target_times])

            # for each forecast file, calculate the anomaly
            ensemble_anomalies = []
            for i, forecast_file in enumerate(atmos_ensemble_files):

                # open file, select var
                forecast = xr.open_dataset(forecast_file)[variable]
                # assert initialization time dimension is only 1. More inits are not supported yet.
                assert len(forecast.time) == 1, "More than 1 initialization time is not supported yet"

                # select target times
                valid_time = forecast.time.values + forecast.step.values
                forecast = forecast.assign_coords({'step': valid_time})
                target_forecast = forecast.sel(step=target_times)
                # collapse singleton time dimension
                target_forecast = target_forecast.isel(time=0).squeeze()
                target_forecast_anomaly = target_forecast - climatology

                # average over time and append to ensemble anomalies
                target_forecast_anomaly = target_forecast_anomaly.mean(dim=('step', 'dayofyear')).squeeze()
                ensemble_anomalies.append(target_forecast_anomaly)

                # plot individual membes 
                if config.plot_individual_members:
                    _plot(
                        anom = target_forecast_anomaly,
                        output_dir = config.output_directory,
                        output_file = f'{variable}-anom_{i:02d}.png',
                        var_name = variable,
                    )
            
            # plot ensemble mean
            ensemble_mean_anomalies = xr.concat(ensemble_anomalies, dim='ensemble').mean(dim='ensemble')
            _plot(
                anom = ensemble_mean_anomalies,
                output_dir = config.output_directory,
                output_file = f'{variable}-anom_ensemble_mean.png',
                var_name = variable,
            )

            # plot observed anomalies
            verif_dataset = xr.open_zarr(config.atmos_verification_file).targets.sel(channel_out=variable)
            observed = verif_dataset.sel(time=target_times)
            observed = observed.assign_coords(time=pd.to_datetime(observed.time).dayofyear).rename({"time": "dayofyear"})
            observed_anomalies = observed - climatology
            # plot these 
            _plot(
                anom = observed_anomalies.mean(dim='dayofyear'),
                output_dir = config.output_directory,
                output_file = f'{variable}-anom_observed.png',
                var_name = variable,
            )

if __name__ == "__main__":

    # receive arguments 
    import argparse
    parser = argparse.ArgumentParser(description="Run skill score comparison between forecasts.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # run analyses 
    main(args.config)