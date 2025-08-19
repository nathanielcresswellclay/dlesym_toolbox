import os
import sys
import time
import logging
from pprint import pprint
from omegaconf import OmegaConf
import numpy as np
import xarray as xr
from toolbox_utils import setup_logging
import matplotlib.pyplot as plt

def get_checkpoint_from_filename(filename: str) -> str:
    """
    Extracts the checkpoint name from the filename.
    Assumes the filename is in the format 'forecast_forced_60day_<checkpoint>.nc'.
    """
    base_name = os.path.basename(filename)
    if base_name.startswith("forecast_forced_60day_") and base_name.endswith(".nc"):
        return base_name[len("forecast_forced_60day_"):-len(".nc")]
    else:
        raise ValueError(f"Filename {filename} does not match expected format.")
    
def main(config_path: str):

    """
    This script runs skill score comparison between different checkpoitns of the same model
    """

    # load the config file
    config = OmegaConf.load(config_path)
    log_file, logger = setup_logging(config.output_log)

    logger.info("Loaded config:\n\n" + OmegaConf.to_yaml(config))  # prints the loaded YAML for clarity
    
    try: 
        from score.rmse_acc import plot_baseline_metrics
    except ImportError as e:
        logger.error("Failed to import required modules. Check mounted modules")
        logger.error(str(e))
        return
    
    # we will call scoring utility for each of the variables
    for var in config.variables:
        logger.info(f"Scoring variable: {var}")
        
        # populate forecast parameters for each model forecast
        # in our config forecast params we list some but not all
        # of the parameters that we need to pass to the scoring utility.
        forecast_params = []
        if config.checkpoints is None:
            # if no checkpoints are specified, use all available forecasts
            logger.info("No checkpoints specified, using all available forecasts.")
            file_list = os.listdir(config.model_path + "/forecasts/60day_forced_checkpoint_ablation")
            checkpoint_forecasts = [os.path.join(config.model_path, "forecasts", "60day_forced_checkpoint_ablation", f) for f in file_list if f.endswith('.nc')]
        else:
            # if checkpoints are specified, use those
            logger.info(f"Using specified checkpoints: {config.checkpoints}")
            checkpoint_forecasts = [os.path.join(config.model_path, "forecasts", "60day_forced_checkpoint_ablation", f"forecast_forced_60day_{c.replace('.mdlus', '')}.nc") for c in config.checkpoints]
        print(f"Using checkpoints: {checkpoint_forecasts}")

        # iterate over each forecast file and create a forecast parameter object
        for forecast_file in checkpoint_forecasts:

            # create a forecast parameter object
            forecast_param = OmegaConf.create() 

            # resolve and add missing parameters for each model
            forecast_param.file = forecast_file
            forecast_param.plot_kwargs = dict(label=get_checkpoint_from_filename(forecast_file))

            forecast_param.verification_file = config.verification_file # shared verification file
            # climatology file will be associated with verification file. Climos are saved variable-wise
            forecast_param.climatology_file = config.verification_file.replace('.zarr', f'_{var}-clima.nc') 
            # rmse and acc caches are assocated with forecast file, again saved variable-wise
            forecast_param.rmse_cache = forecast_param.file.replace('.nc', f'_{var}-rmse.nc')
            forecast_param.acc_cache = forecast_param.file.replace('.nc', f'_{var}-acc.nc')

            # add the forecast parameter to the list
            forecast_params.append(forecast_param)

        print(config.climo_cache_prefix)
        if getattr(config, 'climo_cache_prefix', None) is not None:
            # if climo is requested, add it to the forecast parameters
            climo_param = OmegaConf.create()
            climo_param.file = None
            climo_param.plot_kwargs = dict(label='Climo', color='grey', alpha=0.5)
            climo_param.verification_file = config.verification_file
            climo_param.climatology_file = config.verification_file.replace('.zarr', f'_{var}-clima.nc')
            climo_param.rmse_cache = config.climo_cache_prefix + f'_rmse.nc'
            climo_param.acc_cache = config.climo_cache_prefix + f'_acc.nc'
            forecast_params.append(climo_param)
            # assume sqrt 2 cache is also available 
            if os.path.isfile(config.climo_cache_prefix + f'_sqrt2-rmse.nc'):
                climo_sqrt2_param = OmegaConf.create()
                climo_sqrt2_param.file = None
                climo_sqrt2_param.plot_kwargs = dict(label='sqrt(2)*Climo', linestyle='dashed', color='grey', alpha=0.5)
                climo_sqrt2_param.verification_file = config.verification_file
                climo_sqrt2_param.climatology_file = config.verification_file.replace('.zarr', f'_{var}-clima.nc')
                climo_sqrt2_param.rmse_cache = config.climo_cache_prefix + f'_sqrt2-rmse.nc'
                climo_sqrt2_param.acc_cache = config.climo_cache_prefix + f'_sqrt2-acc.nc'
                forecast_params.append(climo_sqrt2_param)

        
        # run the scoring utility with configured params
        plot_baseline_metrics(
            forecast_params = forecast_params,
            variable = var,
            plot_file = config.output_directory + f"/skill_comp_{config.plot_file_suffix}_{var}.png",
            xlim = config.xlim,
            rmse_ylim = config.rmse_ylim if 'rmse_ylim' in config else None,
            overwrite_cache = config.overwrite_cache,
        )
    
    logger.info("Skill score comparison completed successfully.")

    logger.info(f"Metrics on RMSEs.")

    # loop through each checkpoint forecast and calculate the slope of the rmse after 30 days
    rmse_30d_60d_slope = []
    rmse_0d_5d_mean = []
    if 'climo_cache_prefix' in config:
        rmse_cross_climo = []
        climo_threshold = xr.open_dataarray(config.climo_cache_prefix + f'_rmse.nc').values.mean().item()
        rmse_30d_60d_sqrt2_climo_rmse = []
        sqrt2_climo_threshold = xr.open_dataarray(config.climo_cache_prefix + f'_sqrt2-rmse.nc').values.mean().item()
    rmse_caches = [f.rmse_cache for f in forecast_params if f.file is not None]
    for i, rmse_cache in enumerate(rmse_caches):

        # slope of the RMSE after 30 days
        rmse_30d_60d = xr.open_dataarray(rmse_cache).sel(step=slice(np.timedelta64(30,'D').astype('timedelta64[ns]'), np.timedelta64(60,'D').astype('timedelta64[ns]')))
        slope = np.polyfit(rmse_30d_60d['step'].values.astype(float), rmse_30d_60d.values, 1)[0]
        rmse_30d_60d_slope.append({
            'label': get_checkpoint_from_filename(forecast_params[i].file),
            'metric': slope
        })

        # mean RMSE over the first 5 days
        rmse_0d_5d = xr.open_dataarray(rmse_cache).sel(step=slice(np.timedelta64(0,'D').astype('timedelta64[ns]'), np.timedelta64(5,'D').astype('timedelta64[ns]')))
        mean_rmse = rmse_0d_5d.mean().item()
        rmse_0d_5d_mean.append({
            'label': get_checkpoint_from_filename(forecast_params[i].file),
            'metric': mean_rmse
        })

        #  model RMSE crosses climatology rmse
        if 'climo_cache_prefix' in config:
            rmse = xr.open_dataarray(rmse_cache)
            climo_diff = rmse - climo_threshold
            # find largest step with where climo_diff is negative
            skillful_steps = climo_diff[np.where(climo_diff < 0)[0]]
            rmse_cross_climo.append({
                'label': get_checkpoint_from_filename(forecast_params[i].file),
                'metric': skillful_steps['step'].values[-1] if len(skillful_steps) > 0 else np.timedelta64(0, 'ns'),
            })

            # sqrt(2) climo RMSE crosses climatology rmse
            rmse_30d_60d_sqrt2_climo_rmse.append({
                'label': get_checkpoint_from_filename(forecast_params[i].file),
                'metric': np.sqrt((rmse_30d_60d.values - sqrt2_climo_threshold)**2).mean().item()
            })
    
    logger.info("Finished calculating RMSE metrics. Plotting...")

    def plot_metrics_compare(metric_x, metric_y, xlabel, ylabel, labels, output_file, guidelinex = None, guideliney = None):

        print(f'plotting values: {xlabel}: {metric_x}, {ylabel}:{metric_y}')
        plt.figure(figsize=(5, 5))
        for i,label in enumerate(labels):
            plt.scatter(metric_x[i], metric_y[i], label=label, alpha=0.7)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if guidelinex is not None:
            plt.axvline(x=guidelinex, color='red', linestyle='--')
        if guideliney is not None:
            plt.axhline(y=guideliney, color='red', linestyle='--')
        plt.grid()
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    # get the unit conversion if available 
    scale_factor = getattr(config, 'dimension_conversion', 1.0)
    unit= getattr(config, 'units', 'units')  # default to meters if not specified
    # plot slope of RMSE after 30 days vs mean RMSE over the first 5 days
    plot_metrics_compare(
        metric_x=[m['metric']/scale_factor for m in rmse_0d_5d_mean],
        metric_y=[m['metric'] for m in rmse_30d_60d_slope],
        xlabel=f'Mean RMSE 0-5 days ({unit})',
        ylabel='Slope of RMSE 30-60 days',
        labels=[m['label'] for m in rmse_0d_5d_mean],
        output_file=config.output_directory + f"/rmse-slope_vs_mean-rmse-5d.png",
        guideliney=0.0,  # horizontal line at y=0
    )
    # plot slope of RMSE after 30 days vs deviation from climatology RMSE
    if 'climo_cache_prefix' in config:
        plot_metrics_compare(
            metric_x=[m['metric']/scale_factor for m in rmse_30d_60d_sqrt2_climo_rmse],
            metric_y=[m['metric'] for m in rmse_30d_60d_slope],
            xlabel=f'30-60 day RMSE vs sqrt(2) climatology ({unit})',
            ylabel='Slope of RMSE (30-60 days)',
            labels=[m['label'] for m in rmse_30d_60d_slope],
            output_file=config.output_directory + f"/rmse-slope_vs_sqrt2-climo-rmse-30d.png",
            guideliney=0.0,  # horizontal line at y=0
            guidelinex=0.0,  # vertical line at x=0
        )
        # plot RMSE crossing climatology
        plot_metrics_compare(
            # convert nanoseconds to days
            metric_x=[m['metric'] / np.timedelta64(1, 'D')  for m in rmse_cross_climo],
            metric_y=[m['metric'] for m in rmse_30d_60d_slope if m['metric'] is not None],
            xlabel='RMSE crosses climatology (days)',
            ylabel='Slope of RMSE (30-60 days)',
            labels=[m['label'] for m in rmse_cross_climo if m['metric'] is not None],
            output_file=config.output_directory + f"/rmse-crosses-climo_vs_slope-30-60.png",
            guideliney=0.0,  # horizontal line at y=0
        )
        # plot rmse crossing climo vs 0-5 day mean RMSE
        plot_metrics_compare(
            metric_x=[m['metric'] / np.timedelta64(1, 'D')  for m in rmse_cross_climo],
            metric_y=[m['metric'] for m in rmse_0d_5d_mean if m['metric'] is not None],
            xlabel='RMSE crosses climatology (days)',
            ylabel=f'Mean RMSE 0-5 days ({unit})',
            labels=[m['label'] for m in rmse_cross_climo if m['metric'] is not None],
            output_file=config.output_directory + f"/rmse-crosses-climo_vs_mean-rmse-5d.png",
            guideliney=0.0,  # horizontal line at y=0
        )
        # 0-5day mean rmse vs sqrt(2) climo rmse
        plot_metrics_compare(
            metric_x=[m['metric']/scale_factor for m in rmse_0d_5d_mean if m['metric'] is not None],
            metric_y=[m['metric']/scale_factor for m in rmse_30d_60d_sqrt2_climo_rmse if m['metric'] is not None],
            xlabel='Mean RMSE 0-5 days ({unit})',
            ylabel='30-60 day RMSE vs sqrt(2) climatology ({unit})',
            labels=[m['label'] for m in rmse_0d_5d_mean if m['metric'] is not None],
            output_file=config.output_directory + f"/rmse-mean-0-5_vs_sqrt2-climo-rmse-30d.png",
            guideliney=0.0,  # horizontal line at y=0
            guidelinex=0.0,  # vertical line at x=0
        )
    logger.info("Finished plotting RMSE metrics.")






if __name__ == "__main__":

    # receive arguments 
    import argparse
    parser = argparse.ArgumentParser(description="Run skill score comparison between forecasts.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # run analyses 
    main(args.config)