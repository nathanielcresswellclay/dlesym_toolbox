"""
Forced climatology: compute and plot seasonal climatologies for a configurable
list of variables (e.g. NDVI_gapfill, swvl1) from verification and forecast data.
"""
import os
import sys
import logging
import argparse

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
import cartopy.io.shapereader as shpreader

from toolbox_utils import setup_logging


# Default plot specs for known variables. Config can override via variables[].plot_options.
DEFAULT_VARIABLE_SPECS = {
    "NDVI_gapfill": {
        "file_suffix": "ndvi",
        "plot_options": {
            "cmap_climo": "brbg",
            "cmap_diff": "bwr",
            "levels": np.arange(-0.25, 1.0, 0.05),
            "diff_levels": np.arange(-0.4, 0.4, 0.05),
            "diff_ticks": np.arange(-0.4, 0.41, 0.2),
            "cbar_label": "NDVI",
            "mask_ice_sheets": True,
        },
    },
    "swvl1": {
        "file_suffix": "sm",
        "plot_options": {
            "cmap_climo": "blues",
            "cmap_diff": "bwr_r",
            "levels": np.arange(0, 0.801, 0.05),
            "diff_levels": np.arange(-0.2, 0.201, 0.02),
            "diff_ticks": None,
            "cbar_label": "Soil Moisture (m$^3$m$^{-3}$)",
            "mask_ice_sheets": True,
        },
    },
}


def _get_custom_cmap_brbg():
    bwr = cm.get_cmap("BrBG")
    return mcolors.ListedColormap(bwr(np.linspace(0, 1, 256)))


def _get_custom_cmap_blues():
    bwr = cm.get_cmap("Blues")
    new_colors = bwr(np.linspace(0, 1, 256))
    for i in range(0, 30):
        new_colors[i] = mcolors.to_rgba("whitesmoke")
    return mcolors.ListedColormap(new_colors)


def _get_custom_cmap_bwr():
    bwr = cm.get_cmap("bwr")
    new_colors = bwr(np.linspace(0, 1, 256))
    for i in range(112, 147):
        new_colors[i] = mcolors.to_rgba("whitesmoke")
    return mcolors.ListedColormap(new_colors)


def _get_custom_cmap_bwr_r():
    bwr = cm.get_cmap("bwr_r")
    new_colors = bwr(np.linspace(0, 1, 256))
    for i in range(112, 147):
        new_colors[i] = mcolors.to_rgba("whitesmoke")
    return mcolors.ListedColormap(new_colors)


def _get_cmap(name: str, for_diff: bool):
    if for_diff:
        if name == "bwr":
            return _get_custom_cmap_bwr()
        if name == "bwr_r":
            return _get_custom_cmap_bwr_r()
        return _get_custom_cmap_bwr()
    if name == "brbg":
        return _get_custom_cmap_brbg()
    if name == "blues":
        return _get_custom_cmap_blues()
    return _get_custom_cmap_brbg()


def _add_ice_sheet_mask(ax):
    shpfilename = shpreader.natural_earth(
        resolution="110m", category="cultural", name="admin_0_countries"
    )
    reader = shpreader.Reader(shpfilename)
    for country in reader.records():
        if country.attributes["NAME"] in ["Greenland", "Antarctica"]:
            ax.add_geometries(
                [country.geometry],
                crs=ccrs.PlateCarree(),
                facecolor="grey",
                edgecolor="black",
            )


def _plot_global(
    data: xr.DataArray,
    mapper,
    output_file_prefix: str,
    opts: dict,
    diff_plot: bool = False,
):
    """Plot global seasonal data on a Robinson map."""
    cmap_name = opts["cmap_diff"] if diff_plot else opts["cmap_climo"]
    levels = np.asarray(
        opts["diff_levels"] if diff_plot else opts["levels"]
    )
    cbar_label = opts["cbar_label"]
    mask_ice = opts.get("mask_ice_sheets", True)
    diff_ticks = opts.get("diff_ticks")

    cmap = _get_cmap(cmap_name, for_diff=diff_plot)

    for season in data.season.values:
        season_data = data.sel(season=season)
        data_ll = xr.DataArray(
            mapper.hpx2ll(season_data.values),
            dims=["lat", "lon"],
            coords={
                "lat": np.arange(90, -90.1, -1),
                "lon": np.arange(0, 360, 1),
            },
        )

        fig, ax = plt.subplots(
            figsize=(10, 5), subplot_kw={"projection": ccrs.Robinson()}
        )
        ax.set_title(f"{season} Climatology", fontsize=15)
        ax.add_feature(cfeature.OCEAN, zorder=10, facecolor="white")
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax.coastlines()

        data_ll_cyclic, lon_cyclic = add_cyclic_point(
            data_ll, coord=data_ll.lon
        )

        im = ax.contourf(
            lon_cyclic,
            data_ll.lat,
            data_ll_cyclic,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            levels=levels,
            extend="both",
        )

        if mask_ice:
            _add_ice_sheet_mask(ax)

    cbar = plt.colorbar(
        im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6
    )
    if diff_plot and diff_ticks is not None:
        cbar.set_ticks(np.asarray(diff_ticks))
        cbar.set_label(cbar_label, fontsize=12)
        plt.title(f"{season}", fontsize=15)
        plt.savefig(
            f"{output_file_prefix}_{season}.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)


def _load_or_compute_verification_climatology(
    config: DictConfig,
    channel: str,
    file_suffix: str,
    logger: logging.Logger,
):
    """Load or compute seasonal climatology from verification zarr."""
    climatology_file = config.verification_file.replace(
        ".zarr", f"_{file_suffix}-clima.nc"
    )
    if os.path.exists(climatology_file):
        logger.info(f"Loading cached climatology from {climatology_file}")
        ds = xr.open_dataset(climatology_file)
        # may be stored as 'targets' or as the variable name
        if "targets" in ds:
            clim = ds.targets
        else:
            clim = ds[channel]
    else:
        logger.info(
            f"Calculating climatology for {channel} from {config.verification_file}"
        )
        clim = (
            xr.open_zarr(config.verification_file)
            .targets.sel(channel_out=channel)
            .groupby("time.dayofyear")
            .mean(dim="time")
        )
        clim.to_netcdf(climatology_file)
        logger.info(f"Climatology saved to {climatology_file}")

    clim = clim.assign_coords(
        dayofyear=pd.date_range("2000-01-01", "2000-12-31", freq="D")
    ).groupby("dayofyear.season").mean(dim="dayofyear")
    return clim


def _load_or_compute_forecast_climatology(
    forecast_file: str,
    channel: str,
    file_suffix: str,
    logger: logging.Logger,
):
    """Load or compute seasonal climatology from a forecast netCDF."""
    fcst_climatology_file = forecast_file.replace(
        ".nc", f"_{file_suffix}-clima.nc"
    )
    if not os.path.exists(fcst_climatology_file):
        logger.info(
            f"Calculating climatology for {forecast_file} -> {fcst_climatology_file}"
        )
        fcst = xr.open_dataset(forecast_file)[channel]
        valid_time = fcst.time.values + fcst.step.values
        fcst = fcst.assign_coords({"step": valid_time})
        fcst = (
            fcst.squeeze()
            .reset_coords(drop=True)
            .rename({"step": "time"})
        )
        fcst_climatology = fcst.groupby("time.dayofyear").mean(dim="time")
        fcst_climatology.to_netcdf(fcst_climatology_file)

    logger.info(f"Loading cached climatology from {fcst_climatology_file}")
    fcst_climatology = xr.open_dataset(fcst_climatology_file)[channel]
    fcst_climatology = fcst_climatology.assign_coords(
        dayofyear=pd.date_range("2000-01-01", "2000-12-31", freq="D")
    ).groupby("dayofyear.season").mean(dim="dayofyear")
    return fcst_climatology


def _run_climatology_for_variable(
    config: DictConfig,
    channel: str,
    spec: dict,
    mapper,
    logger: logging.Logger,
):
    """Compute and plot climatology and differences for one variable."""
    file_suffix = spec["file_suffix"]
    opts = spec["plot_options"]

    climatology = _load_or_compute_verification_climatology(
        config, channel, file_suffix, logger
    )

    obs_prefix = config.output_directory + f"/obs_{file_suffix}-climatology"
    logger.info(f"Plotting observed climatology to {obs_prefix}*")
    _plot_global(climatology, mapper, obs_prefix, opts)

    for forecast in config.forecast_params:
        fcst_climatology = _load_or_compute_forecast_climatology(
            forecast.file, channel, file_suffix, logger
        )

        fcst_prefix = (
            config.output_directory + f"/{forecast.model_id}_{file_suffix}-climatology"
        )
        logger.info(f"Plotting forecast climatology to {fcst_prefix}*")
        _plot_global(fcst_climatology, mapper, fcst_prefix, opts)

        diff_prefix = (
            config.output_directory
            + f"/{forecast.model_id}_{file_suffix}-climatology-diff"
        )
        _plot_global(
            fcst_climatology - climatology,
            mapper,
            diff_prefix,
            opts,
            diff_plot=True,
        )


def _resolve_variable_specs(config: DictConfig):
    """Build list of (channel, spec) from config. Uses defaults for known channels."""
    variables = getattr(config, "variables", None)
    if not variables:
        # backward compatibility: default to both NDVI and swvl1
        variables = ["NDVI_gapfill", "swvl1"]

    result = []
    for v in variables:
        if isinstance(v, str):
            channel = v
            if channel not in DEFAULT_VARIABLE_SPECS:
                raise ValueError(
                    f"Unknown variable '{channel}'. Known: {list(DEFAULT_VARIABLE_SPECS.keys())}. "
                    "Use a full spec with channel and plot_options for custom variables."
                )
            spec = dict(DEFAULT_VARIABLE_SPECS[channel])
            result.append((channel, spec))
        else:
            channel = v.channel
            spec = dict(
                DEFAULT_VARIABLE_SPECS.get(
                    channel, {"file_suffix": channel, "plot_options": {}}
                )
            )
            if hasattr(v, "file_suffix"):
                spec["file_suffix"] = v.file_suffix
            if hasattr(v, "plot_options"):
                po = dict(v.plot_options) if v.plot_options else {}
                spec["plot_options"] = {**spec["plot_options"], **po}
            result.append((channel, spec))
    return result


def _plot_forced_climatology(config: DictConfig, logger: logging.Logger):
    try:
        import data_processing.remap.healpix as hpx
    except ImportError as e:
        logger.error("Failed to import data_processing.remap.healpix")
        logger.error(str(e))
        return

    mapper = hpx.HEALPixRemap(
        latitudes=181,
        longitudes=360,
        nside=64,
    )

    for channel, spec in _resolve_variable_specs(config):
        logger.info(f"Processing variable: {channel}")
        _run_climatology_for_variable(config, channel, spec, mapper, logger)

    logger.info("Finished plotting climatologies.")


def main(config_path: str):
    """Load config and plot seasonal climatologies for all configured variables."""
    config = OmegaConf.load(config_path)
    _, logger = setup_logging(config.output_log)

    logger.info("Loaded config:\n\n" + OmegaConf.to_yaml(config))

    _plot_forced_climatology(config, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot seasonal climatologies for configurable variables (e.g. NDVI, soil moisture)."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()
    main(args.config)
