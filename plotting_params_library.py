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
        units = 'm$^3$ m$^{-3}$'
    ),
    stl1 = dict(
        cmap = _get_custom_cmap_split('bwr'),
        anomaly_levels = np.arange(-9,10,1),
        anomaly_ticks = np.arange(-8,9,4),
        label = 'Surface Temperature',
        units = 'K'
    ),
    NDVI_gapfill = dict(
        cmap = _get_custom_cmap_split('BrBG'),
        anomaly_levels = np.arange(-.3,.3001,.0375),
        anomaly_ticks = np.arange(-.3,.31,.15),
        label = 'NDVI', 
        units = 'unitless' 
    ),
    t2m = dict(
        cmap = _get_custom_cmap_split('bwr'),
        anomaly_levels = np.arange(-15,16,2),
        anomaly_ticks = np.arange(-15,16,5),
        label = 'Surface Temperature',
        units = 'K'
    ),
    z500 = dict(
        cmap = _get_custom_cmap_split('bwr'),
        anomaly_levels = np.arange(-100,101,10),
        anomaly_ticks = np.arange(-100,101,20),
        label = '500 hPa Geopotential',
        units = 'dam',
        scaling = 1 / 98.1
    )
)