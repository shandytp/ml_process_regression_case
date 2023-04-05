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