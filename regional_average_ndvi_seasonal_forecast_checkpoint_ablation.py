import os
import re
import sys
import logging
import warnings

# analysis
import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import OmegaConf, DictConfig

# plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle

# mounted modules
from toolbox_utils import setup_logging


def _prepare_forecast_params(config: DictConfig):
    """
    Return list of (file_path, checkpoint_id) from forecast_params or by globbing forecast_directory.
    """
    if config.get('forecast_params') is not None and len(config.forecast_params) > 0:
        return [
            (item.file, item.checkpoint_id)
            for item in config.forecast_params
        ]

    prefix = config.get('forecast_filename_prefix', '')
    forecast_files = [
        os.path.join(config.forecast_directory, f)
        for f in os.listdir(config.forecast_directory)
        if f.endswith('.nc') and (prefix == '' or f.startswith(prefix))
    ]
    forecast_files.sort()

    def _get_checkpoint_id(filepath):
        basename = os.path.basename(filepath).replace('.nc', '')
        if prefix:
            basename = basename[len(prefix):] if basename.startswith(prefix) else basename
        # Shorten to epoch-NNNN if present
        match = re.search(r'epoch-(\d+)', basename, re.IGNORECASE)
        if match:
            return f"epoch-{match.group(1)}"
        return basename[:40] if len(basename) > 40 else basename

    return [(f, _get_checkpoint_id(f)) for f in forecast_files]


def main(config_path: str):
    """Plot regionally averaged NDVI from checkpoint ablation forecasts."""
    config = OmegaConf.load(config_path)
    _, logger = setup_logging(config.output_log)

    logger.info("Loaded config:\n\n" + OmegaConf.to_yaml(config))

    forecast_params = _prepare_forecast_params(config)
    if not forecast_params:
        logger.error("No forecast files found. Check forecast_directory and forecast_filename_prefix.")
        return

    config.forecast_params = forecast_params
    _plot_ndvi_checkpoint_ablation(config, logger)
    logger.info("Finished plotting NDVI checkpoint ablation.")


def _plot_ndvi_checkpoint_ablation(config: DictConfig, logger: logging.Logger):
    if config.get('plot_climatology', False):
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

    # build region index from verification
    lat = xr.open_zarr(config.verification_file).lat
    lon = xr.open_zarr(config.verification_file).lon
    lon = (lon + 180) % 360 - 180
    region_index_lat = np.logical_and(lat > config.region_box.south, lat < config.region_box.north)
    region_index_lon = np.logical_and(lon > config.region_box.west, lon < config.region_box.east)
    region_index = region_index_lat & region_index_lon

    if config.get('plot_region_map', False):
        logger.info("Plotting region box")
        _plot_region_box(config, save_path=os.path.join(config.output_directory, 'region_box.png'))

    # load verification (observations)
    logger.info("Calculating regional average NDVI from verification data")
    obs = xr.open_zarr(config.verification_file).targets.sel(
        channel_out='NDVI_gapfill'
    ).sel(time=slice(config.plot_time_start, config.plot_time_end))
    obs = obs.where(region_index.compute(), drop=True).mean(dim=['face', 'height', 'width'])
    obs = obs.resample(time='7D').mean()

    # initialize figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(config.figure_params.title, fontsize=16)

    logger.info("Plotting observed NDVI data")
    ax.plot(obs.time, obs, label='MODIS Observations', color='black', linewidth=2)

    # plot climatology if requested
    if config.get('plot_climatology', False):
        logger.info("Plotting NDVI climatology")
        climatology_region = climatology.where(region_index.compute(), drop=True).mean(
            dim=['face', 'height', 'width']
        )
        plot_dates = pd.date_range(config.plot_time_start, config.plot_time_end, freq='7D')
        doy_list = list(plot_dates.dayofyear)
        climo_vals = climatology_region.sel(dayofyear=doy_list, method='nearest').values
        ax.plot(plot_dates, climo_vals, label='Climatology', color='gray', linestyle='--', linewidth=1.5)

    # colormap for checkpoint forecasts (viridis by rank)
    n_fcsts = len(config.forecast_params)
    colors = cm.viridis(np.linspace(0.2, 0.8, n_fcsts)) if n_fcsts > 0 else []

    for i, (filepath, checkpoint_id) in enumerate(config.forecast_params):
        logger.info(f"Processing forecast {checkpoint_id}: {filepath}")
        fcst = xr.open_dataset(filepath).NDVI_gapfill

        if getattr(config, 'bias_cache', None) is not None:
            logger.info(f"Applying bias correction using bias cache {config.bias_cache}")
            bias = xr.open_dataset(config.bias_cache).NDVI_gapfill
            init_calendar_day = fcst.time.dt.strftime('%m-%d')
            fcst = fcst.assign_coords({'strftime': init_calendar_day})
            fcst = fcst - bias

        fcst = fcst.where(region_index.compute(), drop=True).mean(dim=['face', 'height', 'width'])

        for init in fcst.time.values:
            valid_time = init + fcst.step.values
            fcst_init = fcst.sel(time=init)
            fcst_init = fcst_init.assign_coords({'step': valid_time}).squeeze()
            fcst_init = fcst_init.sel(step=slice(config.plot_time_start, config.plot_time_end))
            fcst_init = fcst_init.resample(step='7D').mean()

            color = colors[i] if i < len(colors) else 'lightseagreen'
            ax.plot(fcst_init.step, fcst_init, label=checkpoint_id, linewidth=1.5, color=color)

    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('NDVI', fontsize=14)
    # set yrange bewteen .25 and .7 
    ax.set_ylim(.25, .7)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    output_file = os.path.join(config.output_directory, 'regional_average_ndvi_checkpoint_ablation.png')
    logger.info(f"Saving plot to {output_file}")
    fig.savefig(output_file, bbox_inches='tight', **config.figure_params.savefig_params)


def _plot_region_box(config, save_path=None):
    """Plot a world map with the region box defined in config.region_box."""
    west = config.region_box.west
    east = config.region_box.east
    south = config.region_box.south
    north = config.region_box.north

    fig, ax = plt.subplots(
        figsize=(10, 5),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    ax.add_feature(cfeature.LAND, facecolor="darkseagreen")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.LAKES, facecolor="lightblue", edgecolor="black", linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)

    ax.set_global()
    ax.gridlines(draw_labels=True, linewidth=0.5, linestyle="--")

    rect = Rectangle(
        (west, south),
        east - west,
        north - south,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        transform=ccrs.PlateCarree(),
        zorder=10
    )
    ax.add_patch(rect)

    ax.set_extent([west - 10, east + 10, south - 10, north + 10], crs=ccrs.PlateCarree())

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Plot regional average NDVI comparing checkpoint ablation forecasts."
    )
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    main(args.config)
