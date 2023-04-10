import pandas as pd
import numpy as np
import helper
import copy

def read_raw_data(config_data_dir: dict, filename: str) -> pd.DataFrame:
    
    # raw_dataset dir
    raw_dataset_dir = config_data_dir["raw_dataset_dir"]
    
    # Variable to store data
    data = pd.read_csv(raw_dataset_dir + filename)

    return data

def check_data(input_data, params):
    input_data = copy.deepcopy(input_data)
    params = copy.deepcopy(params)
    
    # check data types of data
    assert input_data.select_dtypes("int").columns.to_list() == \
        params["int32_columns"], "Error terjadi di kolom int32."
    assert input_data.select_dtypes("float").columns.to_list() == \
        params["float32_columns"], "Error terjadi di kolom float32."
    assert input_data.select_dtypes("object").columns.to_list() == \
        params["object_columns"], "Error terjadi di kolom object."

    # check range of data
    assert input_data["Year"].between(params["range_Year"][0], params["range_Year"][1]).sum() == \
          len(input_data), "Error terjadi di range Year."
    
    # range sales
    assert input_data["NA_Sales"].between(params["range_NA_Sales"][0], params["range_NA_Sales"][1]).sum() == \
        len(input_data), "Error terjadi di range NA_Sales."
    assert input_data["EU_Sales"].between(params["range_EU_Sales"][0], params["range_EU_Sales"][1]).sum() == \
        len(input_data), "Error terjadi di range EU_Sales."
    assert input_data["JP_Sales"].between(params["range_JP_Sales"][0], params["range_JP_Sales"][1]).sum() == \
        len(input_data), "Error terjadi di range JP_Sales."
    assert input_data["Other_Sales"].between(params["range_Other_Sales"][0], params["range_Other_Sales"][1]).sum() == \
        len(input_data), "Error terjadi di range Other_Sales."
    assert input_data["Global_Sales"].between(params["range_Global_Sales"][0], params["range_Global_Sales"][1]).sum() == \
        len(input_data), "Error terjadi di range Global_Sales."
    
    return "Passed data defense"

if __name__ == "__main__":
    # 1. Load config file
    config_data = helper.load_config()

    # 2. Read all raw_dataset
    raw_dataset = read_raw_data(config_data, "vgsales.csv")

    # 3. Save data into pickle
    helper.dump_pickle(
        raw_dataset,
        config_data["raw_dataset_path"]
    )

    # drop all categories cols

    # 4. Handling year value
    raw_dataset["Year"] = raw_dataset["Year"].\
        fillna(-1).astype("int").copy()
    
    # 5. Check data definition
    check_data(raw_dataset, config_data)

    # 6. Splitting train test
    X = raw_dataset[config_data["predictors"]]