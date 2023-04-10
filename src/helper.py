import yaml
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

def timestamp() -> datetime:
    # mengembalikan current date and time
    return datetime.now()

def load_config(param_dir: str) -> dict:
    # Load yaml file
    try:
        with open(param_dir, "r") as file:
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


def load_params(param_dir):
    with open(param_dir, "r") as file:
        params = yaml.safe_load(file)

    return params

# kalo pake versi ini nggk fleksibel
def read_data(data_dir, filename):
    data = pd.read_csv(data_dir + filename)

    return data

def check_data(input_data, params):
    # check data types of data
    assert input_data.select_dtypes("int").columns.to_list() == params["int32_columns"], "Error terjadi di kolom int32."
    assert input_data.select_dtypes("float").columns.to_list() == params["float32_columns"], "Error terjadi di kolom float32."
    assert input_data.select_dtypes("object").columns.to_list() == params["object_columns"], "Error terjadi di kolom object."

    # check range of data
    assert input_data["Year"].between(params["range_Year"][0], params["range_Year"][1]).sum() == len(input_data), "Error terjadi di range Year."
    
    # range sales
    

    return "Passed data defense"


# preprocessing
def nan_detector(set_data):
    set_data = set_data.copy()

    set_data.replace(-1, np.nan, inplace = True)

    return set_data