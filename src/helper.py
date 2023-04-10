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
PRINT_DEBUG = params["print_debug"]

def print_debug(msg: str) -> None:
    # check if user wants to use print or not
    if PRINT_DEBUG == True:
        print(timestamp(), msg)

