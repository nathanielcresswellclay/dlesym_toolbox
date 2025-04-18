import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import logging
import os
from matplotlib import gridspec
from tqdm import tqdm
from hurricane_track import initialize_evaluator

# set logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# usefull metas
variable_metas = {
    't500' : {
        'level' : 500.,
        'era5_name' : 't',
    },
}

def plot_max_t(ax, data, region, hurricane_id, color, label=None):
    """
    Plot the maximum wind speed of a hurricane on a given axis as a function of step

    Parameters:
    - ax: The axis to plot on.
    - data: The dataset containing hurricane data. Defined in step, lat, lon
    - region: The region to plot the hurricane track in lat/lon coordinates.
    - hurricane_id: The ID of the hurricane to plot.
    - color: The color of the wind speed line.
    - label: The label for the line (optional).
    """

    # select the region
    data = data.sel(lat=slice(region[0], region[1]), lon=slice(region[2], region[3]))

    # find the lat and lon values of the maximum wind speed at each step
    logger.info(f"Plotting max wind speed for {hurricane_id}")

    # initialize buffer to hold location of mins 
    max_t = []
    for s in tqdm(data.step):
        max_t.append(data.sel(step=s).values.max())
    
    # loop over the steps and plot the track, connecting adjacent points
    for i in range(len(max_t) - 1):

        # plot the track between the two points
        ax.plot([data.step.values[i], data.step.values[i + 1]], 
            [max_t[i], max_t[i + 1]],
            color=color, linewidth=2, marker="o", label=label if i == 0 else None)

    return ax

def main(
    forecast_files,
    t_verif,
    var_name,
    region,
    forecast_ids,
    forecast_colors,
    module_dir,
    init_time,
    step_slice,
    title,
    xticks,
    t_ticks,
    output_dir,
):

    # make output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # initialize figure with gridspec for consistent axis sizes
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_title(title)

    # analysis for each forecast 
    for i, (forecast_file, forecast_id, forecast_color) in enumerate(zip(forecast_files, forecast_ids, forecast_colors)):
        logger.info(f"Processing {forecast_id} from {forecast_file}")

        # load the dataset select initialization time
        ds = xr.open_dataset(forecast_file).sel(time=init_time)

        # extract t
        if i==0:
            # initialize evaluators
            ev_t = initialize_evaluator(module_dir, forecast_file, t_verif, var_name,)# calculate_verif=True)
            # select the verification for init and step range
            # verif_t = ev_t.verification_da.sel(time=init_time).isel(step=step_slice)
        else:
            ev_t = initialize_evaluator(module_dir, forecast_file, t_verif, var_name)

        # select forecast for init and step range
        t = ev_t.forecast_da.sel(time=init_time).isel(step=step_slice) - 273.15 # convert to celsius

        # # plot verification data
        # if i==0:
        #     ax = plot_max_t(ax, verif_t, region, "ERA5", "black")

        # plot maximum temp
        ax = plot_max_t(ax, t, region, forecast_id, forecast_color, label=forecast_id)

    # style t speed plot
    fig.legend(loc='upper left', fontsize=8, title="Models", title_fontsize=10)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks/(24*3.6e12))
    ax.set_yticks(t_ticks)
    ax.set_xlabel("Leadtime (days)")
    ax.set_ylabel("Temperature (°C)")
    ax.grid()

    # save the figure
    logger.info(f"Saving figure to {output_dir}/hurricane_max_t.png")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/hurricane_analysis.png", dpi=300)

PARAMS_hydrostatic_all_models = {
    "forecast_files": [
        "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_256_128_128/forecast_60d_monthly.nc",
        "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v3_q50/forecast_60d_monthly.nc",
        "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v3_q75/forecast_60d_monthly.nc",
        "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v3_q95/forecast_60d_monthly.nc",
        "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v2/forecast_60d_monthly.nc"
    ],
    "t_verif":'/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_t850.nc',
    "var_name":"t500",
    "region": [40, 10, 360-90, 360-10],
    "forecast_ids": [
        "baseline",
        "Hydrostatic-q50",
        "Hydrostatic-q75",
        "Hydrostatic-q95",
        "hydrostatic-v2"
    ],
    "forecast_colors": [
        "green",
        "blue",
        "orange",
        "red",
        "purple"
    ],
    "module_dir":"/home/disk/brume/nacc/DLESyM",
    "init_time": "2017-08-31T00:00:00",
    "step_slice": slice(9,48),
    "title": "T$_{500}$ Maxima ",
    "xticks": np.arange(9*6*3.6e12, 48*6*3.6e12, 10*6*3.6e12),
    "t_ticks": np.arange(4,-7,-1)[::-1],
    "output_dir": "/home/disk/brume/nacc/hydrostatic_model_eval/hurricane_max_t500_all-models/",
}

PARAMS_hydrostatic = {
    "forecast_files": [
        "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_256_128_128/forecast_60d_monthly.nc",
        "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v2/forecast_60d_monthly.nc"
    ],
    "t_verif":'/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_t850.nc',
    "var_name":"t500",
    "region": [40, 10, 360-90, 360-10],
    "forecast_ids": [
        "baseline",
        "hydrostatic-v2"
    ],
    "forecast_colors": [
        "green",
        "purple"
    ],
    "module_dir":"/home/disk/brume/nacc/DLESyM",
    "init_time": "2017-08-31T00:00:00",
    "step_slice": slice(9,48),
    "title": "T$_{500}$ Maxima ",
    "xticks": np.arange(9*6*3.6e12, 48*6*3.6e12, 10*6*3.6e12),
    "t_ticks": np.arange(4,-7,-1)[::-1],
    "output_dir": "/home/disk/brume/nacc/hydrostatic_model_eval/hurricane_max_t500/",
}

PARAMS_hydrostatic_t250_all_models = {
    "forecast_files": [
        "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_256_128_128/forecast_60d_monthly.nc",
        "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v3_q50/forecast_60d_monthly.nc",
        "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v3_q75/forecast_60d_monthly.nc",
        "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v3_q95/forecast_60d_monthly.nc",
        "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v2/forecast_60d_monthly.nc"
    ],
    "t_verif":'/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_t850.nc',
    "var_name":"t250",
    "region": [40, 10, 360-90, 360-10],
    "forecast_ids": [
        "baseline",
        "Hydrostatic-q50",
        "Hydrostatic-q75",
        "Hydrostatic-q95",
        "hydrostatic-v2"
    ],
    "forecast_colors": [
        "green",
        "blue",
        "orange",
        "red",
        "purple"
    ],
    "module_dir":"/home/disk/brume/nacc/DLESyM",
    "init_time": "2017-08-31T00:00:00",
    "step_slice": slice(9,48),
    "title": "T$_{250}$ Maxima ",
    "xticks": np.arange(9*6*3.6e12, 48*6*3.6e12, 10*6*3.6e12),
    "t_ticks": np.arange(-30,-43,-2)[::-1],
    "output_dir": "/home/disk/brume/nacc/hydrostatic_model_eval/hurricane_max_t250_all-models/",
}

PARAMS_hydrostatic_t250 = {
    "forecast_files": [
        "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_256_128_128/forecast_60d_monthly.nc",
        # "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v3_q50/forecast_60d_monthly.nc",
        # "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v3_q75/forecast_60d_monthly.nc",
        # "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v3_q95/forecast_60d_monthly.nc",
        "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v2/forecast_60d_monthly.nc"
    ],
    "t_verif":'/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_t850.nc',
    "var_name":"t250",
    "region": [40, 10, 360-90, 360-10],
    "forecast_ids": [
        "baseline",
        # "Hydrostatic-q50",
        # "Hydrostatic-q75",
        # "Hydrostatic-q95",
        "hydrostatic-v2"
    ],
    "forecast_colors": [
        "green",
        # "blue",
        # "orange",
        # "red",
        "purple"
    ],
    "module_dir":"/home/disk/brume/nacc/DLESyM",
    "init_time": "2017-08-31T00:00:00",
    "step_slice": slice(9,48),
    "title": "T$_{250}$ Maxima ",
    "xticks": np.arange(9*6*3.6e12, 48*6*3.6e12, 10*6*3.6e12),
    "t_ticks": np.arange(-30,-43,-2)[::-1],
    "output_dir": "/home/disk/brume/nacc/hydrostatic_model_eval/hurricane_max_t250/",
}

if __name__ == "__main__":
    main(**PARAMS_hydrostatic)
    main(**PARAMS_hydrostatic_all_models)
    main(**PARAMS_hydrostatic_t250_all_models)
    main(**PARAMS_hydrostatic_t250)