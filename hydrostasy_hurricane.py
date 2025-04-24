import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from cartopy.util import add_cyclic_point

import sys
from tqdm import tqdm
from hurricane_track import initialize_evaluator

# dictionaries calibrated for visualizing multiple fields in the same plot
default_plot_fields = dict(
    t500=dict(color='black', verif='/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_t850.nc', levels=np.arange(6,-10,-2)[::-1], label="t500", conversion_func=lambda x:x-273.15 ), # still need to download vars from ERA5
    t250=dict(color='green', verif='/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_t850.nc', levels=np.arange(-30,-43,-2)[::-1], label="t250", conversion_func=lambda x:x-273.15 ), # still need to download vars from ERA5
)
ws10m_z100_z700 = dict(
    ws10m=dict(color='hot_r', verif='/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_t850.nc', pcolor_flag=True, vmin=5,vmax=30, label="10m wind speed"),
    z100=dict(color='black', verif='/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_t850.nc', levels=np.arange(162600,170000,250), label="Z$_{100}$", kwargs=dict(linewidths=1.25)),
    z700=dict(color='black', verif='/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_t850.nc', levels=np.arange(21440,32000,250), label="Z$_{700}$", kwargs=dict(linewidths=.35)),
)
ws10_z1000_z700 = dict(
    ws10m=dict(color='hot_r', verif='/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_t850.nc', pcolor_flag=True, vmin=5,vmax=30, label="10m wind speed"),
    z1000=dict(color='black', verif='/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_t850.nc', levels=np.arange(-3000.,3000,250), label="Z$_{1000}$", kwargs=dict(linewidths=.35)),
    z700=dict(color='black', verif='/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_t850.nc', levels=np.arange(21440,32000,250), label="Z$_{700}$", kwargs=dict(linewidths=1.25)),
)
z850_ws10_z250 = dict(
    ws10m=dict(color='hot_r', verif='/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_t850.nc', pcolor_flag=True, vmin=5,vmax=30, label="10m wind speed"),
    z850=dict(color='black', verif='/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_t850.nc', levels=np.arange(7880.,16000,250), label="Z$_{850}$", kwargs=dict(linewidths=.35)),
    z250=dict(color='black', verif='/home/disk/rhodium/dlwp/data/era5/1deg/era5_1950-2022_3h_1deg_t850.nc', levels=np.arange(107000.,110000.,250), label="Z$_{250}$", kwargs=dict(linewidths=1.25)),
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
        fig, ax = plt.subplots(1, 1, figsize=(10, 7), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([-85, -35, 12, 45], crs=ccrs.PlateCarree())
        ax.coastlines(color='grey', linewidth=0.5)
        return fig, ax
    
    # make the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # loop over forecast steps and plot each field
    for i, step in tqdm(enumerate(forecast_das[0].step.values)):
        fig, ax = init_frame()
        labels = []
        contours = []
        color_fills = []
        color_fill_labels = []
        for j, plot_field in enumerate(plot_fields.keys()):

            temp_field = forecast_das[j].isel(step=i)
            if 'conversion_func' in plot_fields[plot_field].keys(): # check for unit conversion function
                temp_field = plot_fields[plot_field]['conversion_func'](temp_field)
            temp_field, lon = add_cyclic_point(temp_field, coord=forecast_das[j].lon)

            # # these print stats for figuring levels
            # print(plot_field)
            # print(temp_field.mean())
            # print(temp_field.min())
            # print(temp_field.max())
            # try:
            #     print(plot_fields[plot_field]['levels'])
            # except KeyError:
            #     print("No levels specified")

            # add the verification field if it exists
            if 'pcolor_flag' in plot_fields[plot_field].keys():
                colors = ax.pcolormesh(lon, forecast_das[j].lat, temp_field, 
                        cmap=plot_fields[plot_field]['color'], 
                        vmin=plot_fields[plot_field]['vmin'], 
                        vmax=plot_fields[plot_field]['vmax'],  
                        shading='auto', transform=ccrs.PlateCarree())
                color_fills.append(colors)
                color_fill_labels.append(plot_fields[plot_field]['label'])
            else:
                im = ax.contour(lon, forecast_das[j].lat, temp_field, 
                        levels=plot_fields[plot_field]['levels'],
                        colors=plot_fields[plot_field]['color'], 
                        linestyles='solid', **plot_fields[plot_field].get('kwargs', {}))
                contours.append(im)
                labels.append(plot_fields[plot_field]['label'])
                # add contour labels
                # ax.clabel(im, inline=True, fontsize=10, fmt='%1.1f', colors=plot_fields[plot_field]['color'])

        plotted_time = forecast_das[0].time.values + step
        lead_time_hours = int(step / 3.6e12)
        plt.title(f"VALID {str(plotted_time)[:13]} [LEADTIME: {lead_time_hours:04d} hours]", fontsize=15)

        # Create the legend
        ax.legend([c.legend_elements()[0][0] for c in contours], labels, loc='upper right', fontsize=15)
        # add colorbar for pcolor plots
        if len(color_fills) > 0:
            for color_fill, color_fill_label in zip(color_fills, color_fill_labels):
                cbar = plt.colorbar(color_fill, ax=ax, orientation='horizontal', pad=0.05, aspect=20, shrink=0.85)
                cbar.set_label(color_fill_label, fontsize=15)
            cbar.ax.tick_params(labelsize=15)

        frame_name = f"forecast_step_{i:04d}.png"
        plt.savefig(os.path.join(output_dir, frame_name), dpi=300)
        plt.close(fig)

        # clear variables to save memory
        del temp_field
        del lon

####### T500 + T250 #######
PARAMS_hydrostat_v2_t500_t250 = {
    'forecast_file' : "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v2/forecast_60d_monthly.nc",
    "plot_time" : "2017-08-31T00:00:00",
    "plot_steps" : slice(9,48),
    "output_dir" : "/home/disk/brume/nacc/hydrostatic_model_eval/hurricane_track_hydrostatic/hydrostatic_frames_dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v2/t500_t250/",
    "module_path" : "/home/disk/brume/nacc/DLESyM_dev",
}
PARAMS_hydrostat_baseline_t500_t250 = {
    'forecast_file' : "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_256_128_128/forecast_60d_monthly.nc",
    "plot_time" : "2017-08-31T00:00:00",
    "plot_steps" : slice(9,48),
    "output_dir" : "/home/disk/brume/nacc/hydrostatic_model_eval/hurricane_track_hydrostatic/hydrostatic_frames_dbl_conv_next_26ch_ws_256_128_128/t500_t250/",
    "module_path" : "/home/disk/brume/nacc/DLESyM_dev",
}
###### WS + Z100 + Z700 #######
PARAMS_hydrostat_v2_ws10_z100_z700 = {
    'forecast_file' : "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v2/forecast_60d_monthly.nc",
    "plot_time" : "2017-08-31T00:00:00",
    "plot_steps" : slice(9,48),
    "output_dir" : "/home/disk/brume/nacc/hydrostatic_model_eval/hurricane_track_hydrostatic/hydrostatic_frames_dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v2/ws10_z100_z700/",
    "module_path" : "/home/disk/brume/nacc/DLESyM_dev",
    "plot_fields" : ws10m_z100_z700,
}
PARAMS_hydrostat_baseline_ws10_z100_z700 = {
    'forecast_file' : "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_256_128_128/forecast_60d_monthly.nc",
    "plot_time" : "2017-08-31T00:00:00",
    "plot_steps" : slice(9,48),
    "output_dir" : "/home/disk/brume/nacc/hydrostatic_model_eval/hurricane_track_hydrostatic/hydrostatic_frames_dbl_conv_next_26ch_ws_256_128_128/ws10_z100_z700/",
    "module_path" : "/home/disk/brume/nacc/DLESyM_dev",
    "plot_fields" : ws10m_z100_z700,
}
###### Z850 + WS850 + Z250 #######
PARAMS_hydrostat_v2_z850_ws10_z250 = {
    'forecast_file' : "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v2/forecast_60d_monthly.nc",
    "plot_time" : "2017-08-31T00:00:00",
    "plot_steps" : slice(9,48),
    "output_dir" : "/home/disk/brume/nacc/hydrostatic_model_eval/hurricane_track_hydrostatic/hydrostatic_frames_dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v2/z850_ws10_z250/",
    "module_path" : "/home/disk/brume/nacc/DLESyM_dev",
    "plot_fields" : z850_ws10_z250,
}
PARAMS_hydrostat_baseline_z850_ws10_z250 = {
    'forecast_file' : "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_256_128_128/forecast_60d_monthly.nc",
    "plot_time" : "2017-08-31T00:00:00",
    "plot_steps" : slice(9,48),
    "output_dir" : "/home/disk/brume/nacc/hydrostatic_model_eval/hurricane_track_hydrostatic/hydrostatic_frames_dbl_conv_next_26ch_ws_256_128_128/z850_ws10_z250/",
    "module_path" : "/home/disk/brume/nacc/DLESyM_dev",
    "plot_fields" : z850_ws10_z250,
}
###### Z1000 + WS10m +Z700
PARAMS_hydrostat_v2_z1000_ws10_z700 = {
    'forecast_file' : "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v2/forecast_60d_monthly.nc",
    "plot_time" : "2017-08-31T00:00:00",
    "plot_steps" : slice(9,48),
    "output_dir" : "/home/disk/brume/nacc/hydrostatic_model_eval/hurricane_track_hydrostatic/hydrostatic_frames_dbl_conv_next_26ch_ws_hydrostatic_256_128_128_v2/z1000_ws10_z700/",
    "module_path" : "/home/disk/brume/nacc/DLESyM_dev",
    "plot_fields" : ws10_z1000_z700,
}
PARAMS_hydrostat_baseline_z1000_ws10_z700 = {
    'forecast_file' : "/home/disk/brass/nacc/forecasts/hydrostatic_models/dbl_conv_next_26ch_ws_256_128_128/forecast_60d_monthly.nc",
    "plot_time" : "2017-08-31T00:00:00",
    "plot_steps" : slice(9,48),
    "output_dir" : "/home/disk/brume/nacc/hydrostatic_model_eval/hurricane_track_hydrostatic/hydrostatic_frames_dbl_conv_next_26ch_ws_256_128_128/z1000_ws10_z700/",
    "module_path" : "/home/disk/brume/nacc/DLESyM_dev",
    "plot_fields" : ws10_z1000_z700,
}

if __name__ == "__main__":
    # hydrostasy_hurricane(**PARAMS_hydrostat_v2_t500_t250)
    # hydrostasy_hurricane(**PARAMS_hydrostat_baseline_t500_t250)

    # hydrostasy_hurricane(**PARAMS_hydrostat_v2_z850_ws850_z250)
    # hydrostasy_hurricane(**PARAMS_hydrostat_baseline_z850_ws850_z250)

    # hydrostasy_hurricane(**PARAMS_hydrostat_v2_ws10_z100_z700)
    # hydrostasy_hurricane(**PARAMS_hydrostat_baseline_ws10_z100_z700)

    # hydrostasy_hurricane(**PARAMS_hydrostat_v2_z1000_ws10_z700)
    hydrostasy_hurricane(**PARAMS_hydrostat_baseline_z1000_ws10_z700)


