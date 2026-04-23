import os
import sys
import time
import logging
import warnings

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


def main(config_path: str):

    """
    Run land / atmos / ocean seasonal climatology plots for coupled forecasts.

    ``forecast_params`` must be a mapping with keys ``land_file``, ``atmos_file``,
    ``ocean_file``, and ``model_id`` (not a list).
    """

    # load the config file
    config = OmegaConf.load(config_path)
    _, logger = setup_logging(config.output_log)

    logger.info("Loaded config:\n\n" + OmegaConf.to_yaml(config))  # prints the loaded YAML for clarity

    fp = config.forecast_params
    mid = fp.model_id

    if config.plot_land:
        from forced_ndvi_climatology import _plot_ndvi_climo
        from forced_sm_climatology import _plot_sm_climo
        land_one = [{"file": fp.land_file, "model_id": mid}]
        ndvi_config = OmegaConf.create()
        ndvi_config.verification_file = config.verification_file_land
        ndvi_config.output_directory = config.output_directory + "/ndvi_climatology/"
        ndvi_config.forecast_params = land_one
        _plot_ndvi_climo(ndvi_config, logger)
        sm_config = OmegaConf.create()
        sm_config.verification_file = config.verification_file_land
        sm_config.output_directory = config.output_directory + "/sm_climatology/"
        sm_config.forecast_params = land_one
        _plot_sm_climo(sm_config, logger)
    if config.plot_atmos:
        from t2m_climatology import _plot_t2m_climo
        from z500_climatology import _plot_z500_climo
        atmos_one = [{"file": fp.atmos_file, "model_id": mid}]
        t2m_config = OmegaConf.create()
        t2m_config.verification_file = config.verification_file_atmos
        t2m_config.output_directory = config.output_directory + "/t2m_climatology/"
        t2m_config.forecast_params = atmos_one
        _plot_t2m_climo(t2m_config, logger)
        z500_config = OmegaConf.create()
        z500_config.verification_file = config.verification_file_atmos
        z500_config.output_directory = config.output_directory + "/z500_climatology/"
        z500_config.forecast_params = atmos_one
        _plot_z500_climo(z500_config, logger)
    if config.plot_ocean:
        from sst_climatology import _plot_sst_climo
        sst_config = OmegaConf.create()
        sst_config.verification_file = config.verification_file_ocean
        sst_config.output_directory = config.output_directory + "/sst_climatology/"
        sst_config.forecast_params = [{"file": fp.ocean_file, "model_id": mid}]
        _plot_sst_climo(sst_config, logger)


if __name__ == "__main__":

    # receive arguments 
    import argparse
    parser = argparse.ArgumentParser(description="Plot coupled climatology of land, ocean, and atmosphere.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # run analyses 
    main(args.config)

    