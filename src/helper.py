import yaml
import numpy as np
import joblib
from datetime import datetime

config_dir = "config/params.yaml"

def timestamp() -> datetime:
    # mengembalikan current date and time
    return datetime.now()

def load_config() -> dict:
    # Load yaml file
    try:
        with open(config_dir, "r") as file:
            config = yaml.safe_load(file)

    except FileNotFoundError:
        raise RuntimeError("Params file not found in path")
    
    return config

def load_pickle(file_path: str):
    # Load and return pickle file
    return joblib.load(file_path)

def dump_pickle(data, file_path: str) -> None:
    # Dump data into pickle file
    joblib.dump(data, file_path)

params = load_config()


# # preprocessing
# def nan_detector(set_data):
#     set_data = set_data.copy()

#     set_data.replace(-1, np.nan, inplace = True)

#     return set_data