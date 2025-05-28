import os
import sys
import time
import yaml
import pprint
import logging
import argparse
import importlib
import subprocess
from queue import Queue
from omegaconf import OmegaConf

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def _run_experiments_parallel(experiment_params, gpus):
    """
    This internal function runs experiments in parallel using available GPUs.
    Experiments should be configured using analysis suite conventions
    """

    queue = Queue()
    for config in experiment_params:
        queue.put(config)

    running = []  # List of tuples: (process, gpu, config_path)

    while not queue.empty() or running:
        # Remove completed processes
        still_running = []
        for p, gpu, config_path in running:
            if p.poll() is None:
                still_running.append((p, gpu, config_path))
            else:
                logger.info(f"Process on GPU {gpu} completed with return code: {p.returncode}")
                try:
                    os.remove(config_path)
                    logger.debug(f"Deleted temporary config file: {config_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {config_path}: {e}")
        running = still_running

        # Launch new processes if GPUs are available
        available_gpus = set(gpus) - {gpu for _, gpu, _ in running}
        for gpu in available_gpus:
            if queue.empty():
                break

            # Get the next experiment config 
            namespace_config = queue.get()
            experiment_name = namespace_config.pop("experiment_name", None)
            config_cache = namespace_config.pop("config_cache", None)
            pythonpath_additions = namespace_config.pop("mounted_modules", None)
            # Resolve interpolations before saving the config
            resolved_dict = OmegaConf.to_container(namespace_config, resolve=True)
            resolved_config = OmegaConf.create(resolved_dict)

            # Save the resolved config
            OmegaConf.save(config=resolved_config, f=config_cache)
            
            # set up the environment for the subprocess
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONPATH"] = os.pathsep.join(pythonpath_additions) + os.pathsep + env.get("PYTHONPATH", "")

            # report location of log file
            logger.info(f"Running experiment {experiment_name} on GPU {gpu}. logging piped to {namespace_config.output_log}")

            # run the experiment
            p = subprocess.Popen(["python", "forecast_experiments/" + experiment_name + '.py', config_cache], env=env)
            running.append((p, gpu, config_cache))

        time.sleep(5)

def analysis_suite(config: str):

    # load the config file
    cfg = OmegaConf.load(config)
    # log config
    logger.info("Loaded config:\n\n"+OmegaConf.to_yaml(cfg))  # prints the loaded YAML for clarity

    # list of dicts required for analyses routines
    analyses_configs = cfg.analyses

    # make output directories, exisit is ok
    os.makedirs(cfg.output_dir, exist_ok=True)
    for analysis in analyses_configs: os.makedirs(analysis.output_subdir, exist_ok=True)

    # first step is to compile a list of parameters for necessary inferences. 
    # we'll run all necessary inference in parallel, and then run the analyses in sequence.
    inference_params = []
    for analysis in analyses_configs:
        inference_params.append(analysis.pop("inference_params", None))
    _run_experiments_parallel(inference_params, cfg.gpus)

    exit()
    # run each analyses in turn 
    for analysis in analyses_configs:

        # get info for importing the analysis function
        analysis_name = analysis.pop("name")
        module_name = analysis.pop("module")
        function_name = analysis.pop("function")

        # try to import the module and function
        try:
            sys.path.insert(0, cfg.toolbox_dir)
            mod = importlib.import_module(f"{module_name}")
            func = getattr(mod, function_name)
            # get the name of the analysisand log
            logger.info(f"Running analysis: {analysis_name}")
            func(**analysis)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load analysis {module_name}.{function_name}: {e}")


    return

def main():

    # receive arguments 
    parser = argparse.ArgumentParser(description="Run the analysis suite with a YAML config file.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # run analyses 
    analysis_suite(args.config)


if __name__ == "__main__":
    main()