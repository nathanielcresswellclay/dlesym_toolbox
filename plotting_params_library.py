import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

# colormap with white as the middle color, coolwarm base
def _get_custom_cmap_split(map_name: str):
    """
    Create a custom colormap with white as the first color.
    """
    # Get the 'coolwarm' colormaps
    map_colors = cm.get_cmap(map_name)
    # Create a new colormap with white as the middle color
    new_colors = map_colors(np.linspace(0, 1, 256))
    for i in range(118,139): new_colors[i] = mcolors.to_rgba('whitesmoke')  # RGBA for white
    new_cmap = mcolors.ListedColormap(new_colors)
    return new_cmap


plotting_param_library = dict(
    swvl1 = dict(
        cmap = _get_custom_cmap_split('BrBG'),
        anomaly_levels = np.arange(-.18,.19,.02),
        anomaly_ticks = np.arange(-.16,.17,.08),
        label = 'Soil Moisture',
        units = 'm$^3$ m$^{-3}$',
        mask_oceans = True,
    ),
    stl1 = dict(
        cmap = _get_custom_cmap_split('bwr'),
        anomaly_levels = np.arange(-9,10,1),
        anomaly_ticks = np.arange(-8,9,4),
        label = 'Surface Temperature',
        units = 'K',
        mask_oceans = True,
    ),
    NDVI_gapfill = dict(
        cmap = _get_custom_cmap_split('BrBG'),
        anomaly_levels = np.arange(-.3,.3001,.0375),
        anomaly_ticks = np.arange(-.3,.31,.15),
        label = 'NDVI', 
        units = 'unitless',
        mask_oceans = True,
    ),
    t2m = dict(
        cmap = _get_custom_cmap_split('bwr'),
        anomaly_levels = np.arange(-13,14,2),
        anomaly_ticks = np.arange(-12,12,6),
        label = 'Surface Temperature',
        units = 'K',
        mask_oceans = True,
    ),
    z500 = dict(
        cmap = _get_custom_cmap_split('bwr'),
        anomaly_levels = np.arange(-15,16,2),
        anomaly_ticks = np.arange(-15,16,5),
        label = '500hPa Geopotential Height',
        units = 'dam',
        scaling = 1 / 98.1,
        mask_oceans = False,
    )
)

# variable metas specfic to RMSE+ACC scoring
rmse_acc_scoring_variable_metas = {
    'z500': {
        'scale_factor':1/9.81, # Scale factor to convert source units to desired plotting physical units, e.g. 1/9.81 for geopotential height to meters
        'units':'m',           # Physical units of the variable to be used in RMSE label, e.g. 'm' for geopotential height
        'level' : 500.,        # Level of the variable in hPa, e.g. 500 for geopotential height
        'era5_name' : 'z',     # Name of the variable in the ERA5 dataset, e.g. 'z' for geopotential height
        'rename_func' : lambda x: x.rename({'geopotential': 'z500'}).drop('level').squeeze(), # Function to reformat era5 dims to match forecast dims
        'yticks' : np.arange(0, 1000, 100), # y-axis ticks to use for RMSE plot
    },
    'z1000': {
        'scale_factor':1/9.81,
        'units':'m',
    },
    'z250': {
        'scale_factor':1/9.81,
        'units':'m',
    },
    't850': {
        'scale_factor':1,
        'units':'C',
        'level' : 850.,
        'era5_name' : 't',
    },
    't2m': {
        'scale_factor':1,
        'units':'C',
        'level' : None,
        'era5_name' : '2m',
    },
    'sst': {
        'scale_factor':1,
        'units':'K',
        'level' : None,
        'era5_name' : 'sst',
    },
    'q2m': {
        'scale_factor': 1000,
        'units':'g/kg',
        'level' : None,
        'era5_name' : '2m',
    },
    'ws10m': {
        'scale_factor':1,
        'units':'m/s',
        'level' : None,
        'era5_name' : '10m',
    },
    'swvl1': {
        'scale_factor':1,
        'units':'m3/m3',
    },
}