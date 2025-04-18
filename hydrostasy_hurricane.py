import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from cartopy.util import add_cyclic_point

import sys
from tqdm import tqdm
from hurricane_track import initialize_evaluator

# default dictionary for visualizing hydrostasy 
default_plot_fields = dict(
    # t850=dict(color='blue', verif='/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_t850.nc', levels=np.arange(25,13,-2)[::-1]),
    t500=dict(color='red', verif='/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_t850.nc', levels=np.arange(6,-10,-2)[::-1], label="t500"), # still need to download vars from ERA5
    t250=dict(color='green', verif='/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_t850.nc', levels=np.arange(-30,-43,-2)[::-1], label="t250"), # still need to download vars from ERA5
)

def hydrostasy_hurricane(forecast_file, plot_time, plot_steps, output_dir, module_path, plot_fields=default_plot_fields):
    """
    Plots overlapping fieds for evaluating hydrostatic coherence.
    Parameters
    ----------
    forecast_file : str
        Path to the forecast file.
    plot_time : str
        List of initialization time to plot. Interpretable as a datetime string. example: '2017-08-31T00:00:00'
    plot_steps : slice of ints
        List of forecast step indices to plot. Interpretable as a slice object. example: slice(9,48)
    output_dir : str
        Path to the output directory.
    module_path : str
        Path to the module directory.
    plot_fields : dict
        Dictionary of fields to plot. The keys are the field names and the values are dictionaries with the following keys:
            - color: str, color of the field
            - verif: str, path to the verification file (optional)
    """
    
    forecast_das = []
    # initialize evaluators around plot fields 
    for plot_field in plot_fields.keys():
            forecast_das.append(initialize_evaluator(module_path, forecast_file, 
                plot_fields[plot_field]['verif'], plot_field, calculate_verif=False).forecast_da.sel(time=plot_time).isel(step=plot_steps))

    

    # helper function to initialize plots 
    def init_frame():
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([-100, -20, 10, 50], crs=ccrs.PlateCarree())
        ax.coastlines()
        return fig, ax
    
    # make the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # loop over forecast steps and plot each field
    for i, step in tqdm(enumerate(forecast_das[0].step.values)):
        fig, ax = init_frame()
        labels = []
        contours = []
        for j, plot_field in enumerate(plot_fields.keys()):

            temp_field = forecast_das[j].isel(step=i) - 273.15 # convert to celsius
            temp_field, lon = add_cyclic_point(temp_field, coord=forecast_das[j].lon)

            # add the verification field if it exists
            im = ax.contour(lon, forecast_das[j].lat, temp_field, 
                       levels=plot_fields[plot_field]['levels'],
                       colors=plot_fields[plot_field]['color'], 
                       linestyles='solid', linewidths=0.5)
            contours.append(im)
            labels.append(plot_fields[plot_field]['label'])

        plotted_time = forecast_das[0].time.values + step
        plt.title(f"VALID {str(plotted_time)[:13]}")

        # Create the legend
        ax.legend([c.legend_elements()[0][0] for c in contours], labels, loc='upper right', fontsize=10)

        frame_name = f"forecast_step_{i:04d}.png"
        plt.savefig(os.path.join(output_dir, frame_name), dpi=300)
        plt.close(fig)

        # clear variables to save memory
        del temp_field
        del lon



PARAMS_hydrostat_v2 = {
    'forecast_file' : "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v2/forecast_60d_monthly.nc",
    "plot_time" : "2017-08-31T00:00:00",
    "plot_steps" : slice(9,48),
    "output_dir" : "/home/disk/brume/nacc/hydrostatic_model_eval/hurricane_track_hydrostatic/hydrostatic_frames_dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v2/",
    "module_path" : "/home/disk/brume/nacc/DLESyM_dev",
}
PARAMS_hydrostat_baseline = {
    'forecast_file' : "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_256_128_128/forecast_60d_monthly.nc",
    "plot_time" : "2017-08-31T00:00:00",
    "plot_steps" : slice(9,48),
    "output_dir" : "/home/disk/brume/nacc/hydrostatic_model_eval/hurricane_track_hydrostatic/hydrostatic_frames_dbl_conv_next_26ch_ws_256_128_128/",
    "module_path" : "/home/disk/brume/nacc/DLESyM_dev",
}

if __name__ == "__main__":

    hydrostasy_hurricane(**PARAMS_hydrostat_v2)
    hydrostasy_hurricane(**PARAMS_hydrostat_baseline)


