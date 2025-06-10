import os
from pathlib import Path


def resolve_output_directory(atmos_model: str, ocean_model: str):
    """
    Resolve the output directory based on the atmospheric and oceanic model names.
    
    Args:
        atmos_model (str): path to the atmospheric model.
        ocean_model (str): path to the oceanic model.
    
    Returns:
        str: The resolved output directory path.
    """

    atmos_file_path = Path(atmos_model)
    ocean_file_path = Path(ocean_model)
    models_dir = atmos_file_path.parent
    dir_name = atmos_file_path.name + "+" + ocean_file_path.name

    return os.path.join(models_dir, dir_name, 'forecasts')