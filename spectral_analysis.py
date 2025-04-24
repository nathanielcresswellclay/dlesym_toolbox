import numpy as np 
import xarray as xr 
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

PARAMS_45N = {
    'forecast_file':'/home/disk/brass/nacc/forecasts/agu_coupled_ensemble/atmos_1152h_hpx32_coupled-dlwp_agu_seed2-best+hpx32_coupled-dlom_agu_seed20_bs128_large-test-best.nc',
    'ref_file':'/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1959-2021_z500.nc',
    'variable':'z500',
    'unit_conversion':1,
    'lat':45,
    'title':'Z$_{500}$ Spectral Power at 45$^{\circ}$N',
    'leadtimes':[pd.Timedelta(d,'D') for d in [0,.25,.5,.75,1,2,14] ],
    'figure_file':'spectral_analysis_45N.pdf',
    'leadtime_plt_params':[
        {'label':'Initialization',
         'color':'black',
         'linewidth':1,
         'alpha':1},
        {'label':'6H',
         'color':'red',
         'linewidth':.75,
         'alpha':1},
        {'label': '12 hours',
         'color':'orange',
         'linewidth':.75,
         'alpha':.8},
        {'label':'18H',
         'color':'cyan',
         'linewidth':.75,
         'alpha':1},
        {'label': '1 day',
         'color':'green',
         'linewidth':.75,
         'alpha':.9},
        {'label':'2 day',
         'color':'blue',
         'linewidth':.75,
         'alpha':.7},
        {'label':'14 day',
         'color':'purple',
         'linewidth':.75,
         'alpha':.7},
    ],
    'ref_line_exp':5/3,
}

def get_closest_lat(target, lats):

    return lats.values[np.unravel_index(np.argmin(np.abs(lats.values-target)),lats.values.shape)] 

def get_lat_band(da, ref_lat, ref_lon, lat):
    

    print(f'Latitude requested was {lat}')
    closest = get_closest_lat(lat, ref_lat)
    boo = ref_lat.values == closest
    print(f'Using closest available lat: {closest} with {boo.sum()} points')
    
    # find the indices that will sort the latitude band by longitude 
    sorted_lon = ref_lon.values[boo].argsort()
    # initialize lat_band array and populate times with lat slices sorted by their lon 
    lat_band = np.empty(da.values.shape[0:2]+(len(sorted_lon),))
    print('Populating latitude band from HEALPix data...')
    times = tqdm(range(lat_band.shape[0]))
    steps = tqdm(range(lat_band.shape[1]),leave=False)
    for t in times:
        times.set_description(f'{str(da.time.values[t])[:13]}')
        for s in steps:
            steps.set_description(f'{str(s)}')
            lat_band[t,s,:] = da.values[t,s,:,:][boo][sorted_lon]
 
    # create dataarray to return 
    lat_band_da = xr.DataArray(
        data = lat_band,    
        dims = ['time','step','lon'],
        coords = dict(
            time=(['time'], da.time.values),
            step=(['step'], da.step.values),
            lon=(['lon'], ref_lon.values[boo][sorted_lon])
        ),
        attrs=dict(
            latitude=closest
        ))
    return lat_band_da

def get_nondimensional_wn(lon,lat):
    
    dx = (111.321*np.cos((np.pi/180)*lat))*np.abs(lon[0]-lon[1])
    return np.fft.rfftfreq(len(lon),dx)*len(lon)*dx

def get_zonal_wavelength(lon,lat):
    
    dx = (111.321*np.cos((np.pi/180)*lat))*np.abs(lon[0]-lon[1])
    return (len(lon)*dx)/get_nondimensional_wn(lon,lat)

def normalize_spectra(spectra, rectangular, lat):
 
    # normalization routine used to satisfy Parseval's relation 
    # taken from Durran et al. 2017 equation 13
    

    # Enforce Parseval's relation of transform unity and return.
    dx = (np.cos(np.deg2rad(lat))*111321)*np.mean(rectangular.lon[1:].values-rectangular.lon[:-1].values)
    N = len(rectangular.lon)
    kd = np.ones(len(spectra))
    kd[-1]=0
    return (dx/(np.pi*N*(1+kd)))*spectra**2

def get_spectra(lat_band, check_parseval_sat=False):
   
    spectra = []
    for t in lat_band.time:
        spectra.append(normalize_spectra(np.abs(np.fft.rfft(lat_band.sel(time=t))),lat_band.sel(time=t),lat_band.latitude))
    
    # check parseval's theorem
    if check_parseval_sat:
    # this expression of Parseval's relation is taken from Durran et al. 2017 equation 11 
        print('Departure from absolute satisfaction of Parsevals Theorem by leadtime:')
        dx = ((np.cos(np.deg2rad(lat_band.latitude))*111321)*(lat_band.lon[1]-lat_band.lon[0])).values
        L = dx*len(lat_band.isel(time=0))
        dk = 2*np.pi/L
        print(1-((1/L)*(dx*(lat_band.values)**2).sum(axis=1)/((np.array(spectra))*dk).sum(axis=1)))
    return np.array(spectra)

def main(params):

    """
    Apply spectral profiles of a forecast at various leadtimes along a latitude band.

    params: dict: containes all parameters needed to run the analysis.
        params['forecast_file']: str: path to the forecast file
        params['ref_file']: str: path to the verification
        params['variable']: str: variable to be analyzed
        params['unit_conversion']: float: unit conversion factor
        params['lat']: float: latitude band to select
        params['title']: str: title of the plot
        params['leadtimes']: list: list of leadtimes over which to calculate spectra
        params['figure_file']: str: path to save the figure
        params['leadtime_plt_params']: list: list of dictionaries containing plot parameters for each leadtime
    """
    
    # open forecast file and extract latitude band 
    fcst = xr.open_dataset(params['forecast_file'])[params['variable']]/params['unit_conversion']
    ref_lat = xr.open_dataset(params['ref_file'])['lat']
    ref_lon = xr.open_dataset(params['ref_file'])['lon']
    lat_band = get_lat_band(fcst, ref_lat, ref_lon, params['lat'])
    xr.set_options(keep_attrs=True)
    # convert to height
    lat_band = lat_band/9.81
    
    fig, ax = plt.subplots(figsize=(5,5))
    wavelength = get_zonal_wavelength(lat_band.lon.values,get_closest_lat(params['lat'],ref_lat))
    for i,lt in enumerate(params['leadtimes']):
        if isinstance(lt,int):
            mean_spectra = get_spectra(lat_band.isel(step=lt)).mean(axis=0)
        else:
            mean_spectra = get_spectra(lat_band.sel(step=lt)).mean(axis=0)
        ax.plot(wavelength[:-1], mean_spectra[:-1], **params['leadtime_plt_params'][i]) 
    # plot reference line 
    #ax.plot(wavelength,wavelength**params['ref_line_exp'],alpha=.5,color='black',linestyle='dotted')
     
    # set up axis and labels 
    ax.invert_xaxis() 
    ax.grid(which='both')
    ax.set_xscale('log')
    ax.set_xlabel('Wavelength (km)', fontsize=12)
    
    ax.set_yscale('log')
    ax.set_ylabel('Power Spectral Density (m$^{3}$)',fontsize=12)
    ax.set_title(params['title'])
    fig.legend(loc=(.18,.14))
    fig.tight_layout()
    # save figure 
    fig.savefig(params['figure_file'], dpi=300)

if __name__ == "__main__":
    
    main(PARAMS_45N)
