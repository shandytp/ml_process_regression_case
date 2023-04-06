import yaml
import pandas as pd


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

    return "Passed data defense"