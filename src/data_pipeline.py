import pandas as pd
import numpy as np
import helper
import copy
from sklearn.model_selection import train_test_split

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
    assert input_data.select_dtypes("float").columns.to_list() == \
        params["float32_columns"], "Error terjadi di kolom float32."

    # check range sales data
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
    print("======= START DATA PIPELINE PROCESS =======")

    # 1. Load config file
    config_data = helper.load_config()

    # 2. Read all raw_dataset
    raw_dataset = read_raw_data(config_data, "vgsales.csv")

    # 3. Save raw data into pickle
    helper.dump_pickle(
        raw_dataset,
        config_data["raw_dataset_path"]
    )

    # 4. Handling unused cols
    raw_dataset = raw_dataset.drop(
        config_data["cols_drop"],
        axis = 1
    )

    # 5. Save cleaned raw data into pickle
    helper.dump_pickle(
        raw_dataset,
        config_data["cleaned_raw_dataset_path"]
    )
    
    # 6. Check data definition
    check_data(raw_dataset, config_data)

    # 7. Splitting input output
    X = raw_dataset[config_data["predictors"]].copy()
    y = raw_dataset[config_data["label"]].copy()

    # 8. Splitting train test
    X_train, X_test, \
    y_train, y_test = train_test_split(
        X, y,
        test_size = 0.2,
        random_state = 42
    )

    # 9. Splitting valid test
    X_valid, X_test, \
    y_valid, y_test = train_test_split(
        X_test, y_test,
        test_size = 0.5,
        random_state = 42
    )

    # 10. Save train, valid, and test set
    helper.dump_pickle(X_train, config_data["train_set_path"][0])
    helper.dump_pickle(y_train, config_data["train_set_path"][1])

    helper.dump_pickle(X_valid, config_data["valid_set_path"][0])
    helper.dump_pickle(y_valid, config_data["valid_set_path"][1])

    helper.dump_pickle(X_test, config_data["test_set_path"][0])
    helper.dump_pickle(y_test, config_data["test_set_path"][1])

    print("======= END DATA PIPELINE PROCESS =======")

