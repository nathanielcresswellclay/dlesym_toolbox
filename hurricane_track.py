import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import logging
import os
from matplotlib import gridspec
from tqdm import tqdm

# set logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# usefull metas
variable_metas = {
    'z500' : {
        'level' : 500.,
        'era5_name' : 'z',
        'rename_func' : lambda x: x.rename({'geopotential': 'z500'}).drop('level').squeeze(),
    },
    'z1000' : {
        'level' : 1000.,
        'era5_name' : 'z',
        'rename_func' : lambda x: x.rename({'geopotential': 'z1000'}).drop('level').squeeze(),
    },
    't850' : {
        'level' : 850.,
        'era5_name' : 't',
    },
    't2m0' : {
        'level' : None,
        'era5_name' : '2m',
    },
    'sst' : {
        'level' : None,
        'era5_name' : 'sst',
    },
    'ws10m' : {
        'level' : None,
        'era5_name' : 'ws10',
        'rename_func' : lambda x: x.rename({'ws10': 'ws10m'}).drop('level').squeeze(),
    },
}

def plot_hurricane_track(ax, data, region, hurricane_id, color):
    """
    Plot the track of a hurricane on a given axis.

    Parameters:
    - ax: The axis to plot on.
    - data: The dataset containing hurricane data. Defined in step, lat, lon
    - region: The region to plot the hurricane track in lat/lon coordinates.
    - hurricane_id: The ID of the hurricane to plot.
    - color: The color of the track line.
    """

    # select the region
    data = data.sel(lat=slice(region[0], region[1]), lon=slice(region[2], region[3]))

    # find the lat and lon values of the minimum z1000 at each step
    logger.info(f"Plotting hurricane track for {hurricane_id}")

    # initialize buffer to hold location of mins 
    min_lats = []
    min_lons = []
    for s in tqdm(data.step):
        # select the data for the current step
        step_data = data.sel(step=s)

        # Find the index of the minimum value in the 2D array
        min_idx = np.unravel_index(np.argmin(step_data.values), step_data.values.shape)

        # Extract the corresponding latitude and longitude
        min_lats.append(step_data.lat[min_idx[0]].values)
        min_lons.append(step_data.lon[min_idx[1]].values)
    
    # loop over the steps and plot the track, connecting adjacent points
    for i in range(len(min_lats) - 1):

        # plot the track between the two points
        ax.plot([min_lons[i], min_lons[i + 1]],
            [min_lats[i], min_lats[i + 1]], 
            color=color, linewidth=2, marker="o", transform=ccrs.PlateCarree(),
            label=hurricane_id if i == 0 else "")

    return ax

def plot_min_z1000(ax, data, region, hurricane_id, color):
    """
    Plot the minimum sea level pressure of a hurricane on a given axis as a function of step

    Parameters:
    - ax: The axis to plot on.
    - data: The dataset containing hurricane data. Defined in step, lat, lon
    - region: The region to plot the hurricane track in lat/lon coordinates.
    - hurricane_id: The ID of the hurricane to plot.
    - color: The color of the pressure line.
    """
    """
    Plot the track of a hurricane on a given axis.

    Parameters:
    - ax: The axis to plot on.
    - data: The dataset containing hurricane data. Defined in step, lat, lon
    - region: The region to plot the hurricane track in lat/lon coordinates.
    - hurricane_id: The ID of the hurricane to plot.
    - color: The color of the track line.
    """

    # select the region
    data = data.sel(lat=slice(region[0], region[1]), lon=slice(region[2], region[3]))

    # find the lat and lon values of the minimum z1000 at each step
    logger.info(f"Plotting min z1000 for {hurricane_id}")

    # initialize buffer to hold location of mins 
    min_z1000 = []
    for s in tqdm(data.step):
        min_z1000.append(data.sel(step=s).values.min())
    
    # loop over the steps and plot the track, connecting adjacent points
    for i in range(len(min_z1000) - 1):

        # plot the track between the two points
        ax.plot([data.step.values[i], data.step.values[i + 1]], 
            [min_z1000[i], min_z1000[i + 1]],
            color=color, linewidth=2, marker="o")

    return ax

def plot_max_ws(ax, data, region, hurricane_id, color):
    """
    Plot the maximum wind speed of a hurricane on a given axis as a function of step

    Parameters:
    - ax: The axis to plot on.
    - data: The dataset containing hurricane data. Defined in step, lat, lon
    - region: The region to plot the hurricane track in lat/lon coordinates.
    - hurricane_id: The ID of the hurricane to plot.
    - color: The color of the wind speed line.
    """

    # select the region
    data = data.sel(lat=slice(region[0], region[1]), lon=slice(region[2], region[3]))

    # find the lat and lon values of the maximum wind speed at each step
    logger.info(f"Plotting max wind speed for {hurricane_id}")

    # initialize buffer to hold location of mins 
    max_ws = []
    for s in tqdm(data.step):
        max_ws.append(data.sel(step=s).values.max())
    
    # loop over the steps and plot the track, connecting adjacent points
    for i in range(len(max_ws) - 1):

        # plot the track between the two points
        ax.plot([data.step.values[i], data.step.values[i + 1]], 
            [max_ws[i], max_ws[i + 1]],
            color=color, linewidth=2, marker="o")

    return ax

def initialize_evaluator(module_dir, forecast_file, verif_file, var, calculate_verif=False):
    """
    Initialize the evaluator for a given variable within the forecast 
    Parameters:
    - module_dir: The directory containing the module.
    - forecast_file: The path to the forecast file.
    - verif_file: The path to the verification file.
    - var: The variable to evaluate.
    - calculate_verif: Whether to calculate verification data.
    """

    # import the evaluator module
    import sys
    sys.path.append(module_dir)
    try:
        from evaluation import evaluators as ev
    except ImportError:
        logger.error(f"Failed to import evaluator module from {module_dir}")
        raise ValueError(f"Failed to import evaluator module from {module_dir}")

    #  initialize evaluator, remapping forecast 
    forecast_ev = ev.EvaluatorHPX(
        forecast_path = forecast_file,
        verification_path = verif_file,
        eval_variable = var,
        on_latlon = True,
        poolsize = 20,
        ll_file=forecast_file.replace('.nc',f'_{var}_ll.nc')
    )
    if calculate_verif:
        forecast_ev.generate_verification(
            verification_path = verif_file,
        )
    return forecast_ev

def main(
    forecast_files,
    z1000_verif,
    ws_verif,
    forecast_ids,
    forecast_colors,
    region,
    module_dir,
    init_time,
    step_slice,
    plot_extent,
    xticks,
    z1000_ticks,
    ws_ticks,
    output_dir,
):

    # make output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # initialize figure with gridspec for consistent axis sizes
    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 1])  # Equal widths for all axes

    # initialize axes
    ax = [None] * 3
    ax[0] = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())  # Only ax[0] gets the projection
    ax[1] = fig.add_subplot(gs[1])  # Regular 2D plot
    ax[2] = fig.add_subplot(gs[2])  # Regular 2D plot


    # format hurricane track plot
    ax[0].set_title("Hurricane Track")
    ax[0].set_extent(plot_extent, crs=ccrs.PlateCarree())
    ax[0].coastlines()

    # format minimum sea level pressure plot
    ax[1].set_title("Minimum $Z_{1000}$")

    # format maximum wind speed plot
    ax[2].set_title("Maximum Wind Speed")

    # analysis for each forecast 
    for i, (forecast_file, forecast_id, forecast_color) in enumerate(zip(forecast_files, forecast_ids, forecast_colors)):
        logger.info(f"Processing {forecast_id} from {forecast_file}")

        # load the dataset select initialization time
        ds = xr.open_dataset(forecast_file).sel(time=init_time)

        # extract z1000 and ws variables
        if i==0:
            # initialize evaluators
            ev_z1000 = initialize_evaluator(module_dir, forecast_file, z1000_verif, "z1000", calculate_verif=True)
            ev_ws = initialize_evaluator(module_dir, forecast_file, ws_verif, "ws10m", calculate_verif=True)
            # select the verification for init and step range
            verif_z1000 = ev_z1000.verification_da.sel(time=init_time).isel(step=step_slice) / 9.81
            verif_ws = ev_ws.verification_da.sel(time=init_time).isel(step=step_slice)
        else:
            ev_z1000 = initialize_evaluator(module_dir, forecast_file, z1000_verif, "z1000")
            ev_ws = initialize_evaluator(module_dir, forecast_file, ws_verif, "ws10m")

        # select forecast for init and step range
        z1000 = ev_z1000.forecast_da.sel(time=init_time).isel(step=step_slice) / 9.81 # convert to m
        ws = ev_ws.forecast_da.sel(time=init_time).isel(step=step_slice)

        # plot verification data
        if i==0:
            # plot hurricane track
            ax[0] = plot_hurricane_track(ax[0], verif_z1000, region, "ERA5", "black")
            # plot minimum z1000
            ax[1] = plot_min_z1000(ax[1], verif_z1000, region, "ERA5", "black")
            # plot maximum wind speed
            ax[2] = plot_max_ws(ax[2], verif_ws, region, "ERA5", "black")

        # plot hurricane track
        ax[0] = plot_hurricane_track(ax[0], z1000, region, forecast_id, forecast_color)
        # plot minimum z1000
        ax[1] = plot_min_z1000(ax[1], z1000, region, forecast_id, forecast_color)
        # # plot maximum wind speed
        ax[2] = plot_max_ws(ax[2], ws, region, forecast_id, forecast_color)

    # style hurricane tracks 
    ax[0].legend(loc='upper right', fontsize=10, frameon=False)
    
    # style min z1000 plot 
    ax[1].set_xticks(xticks)
    ax[1].set_xticklabels(xticks/(24*3.6e12))
    ax[1].set_xlabel("Leadtime (days)")
    ax[1].set_yticks(z1000_ticks)
    ax[1].set_ylabel("$Z_{1000}$ (m)")
    ax[1].grid()

    # style max wind speed plot
    ax[2].set_xticks(xticks)
    ax[2].set_xticklabels(xticks/(24*3.6e12))
    ax[2].set_yticks(ws_ticks)
    ax[2].set_xlabel("Leadtime (days)")
    ax[2].set_ylabel("Wind Speed (m/s)")
    ax[2].grid()

    # save the figure
    logger.info(f"Saving figure to {output_dir}/hurricane_analysis.png")
    # plt.tight_layout()
    plt.savefig(f"{output_dir}/hurricane_analysis.png", dpi=300)

PARAMS_hydrostatic = {
    "forecast_files": [
        "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_256_128_128/forecast_60d_monthly.nc",
        # "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v3_q50/forecast_60d_monthly.nc",
        # "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v3_q75/forecast_60d_monthly.nc",
        # "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v3_q95/forecast_60d_monthly.nc",
        "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v2/forecast_60d_monthly.nc"
    ],
    "z1000_verif":"/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_z1000.nc",
    "ws_verif":"/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_windspeed.nc",
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
    "region": [40, 10, 360-90, 360-10],
    "module_dir":"/home/disk/brume/nacc/DLESyM",
    "init_time": "2017-08-31T00:00:00",
    "step_slice": slice(9,48),
    "plot_extent": [-90, -30, 10, 50],
    "xticks": np.arange(9*6*3.6e12, 48*6*3.6e12, 10*6*3.6e12),
    "z1000_ticks": np.arange(-310,55,50),
    "ws_ticks": np.arange(12, 33, 2),
    "output_dir": "/home/disk/brume/nacc/hydrostatic_model_eval/hurricane_track_hydrostatic",
}

if __name__ == "__main__":
    main(**PARAMS_hydrostatic)