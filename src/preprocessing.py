import numpy as np
import pandas as pd
import helper

def load_dataset(config_data: dict) -> pd.DataFrame:
    # Load every data that have we created
    X_train = helper.load_pickle(config_data["train_set_path"][0])
    y_train = helper.load_pickle(config_data["train_set_path"][1])

    X_valid = helper.load_pickle(config_data["valid_set_path"][0])
    y_valid = helper.load_pickle(config_data["valid_set_path"][1])

    X_test = helper.load_pickle(config_data["test_set_path"][0])
    y_test = helper.load_pickle(config_data["test_set_path"][1])

    # Concat X and y for each set
    train_set = pd.concat(
        [X_train, y_train],
        axis = 1
    )

    valid_set = pd.concat(
        [X_valid, y_valid],
        axis = 1    
    )

    test_set = pd.concat(
        [X_test, y_test], 
        axis = 1 
    )

    # return three set of data that created

    return train_set, valid_set, test_set

def nan_detector(set_data: pd.DataFrame, params):
    
    # copy data
    set_data = set_data.copy()

    # create temp list for save .isna() value
    tmp_res = []

    # looping for all cols
    for col in params["float32_columns"]:
        count = set_data[col].isnull().sum()
        tmp_res.append(count)

    assert sum(tmp_res) == 0, "Terdapat missing values pada data."

    return "Proses nan detector telah selesai"

# tidak wajib
def log_transform(set_data: pd.DataFrame, params) -> pd.DataFrame:
    
    # copy data
    set_data = set_data.copy()

    for col in params["predictors"]:
        # transform using log
        set_data[col] = np.log(set_data[col])

    return set_data

if __name__ == "__main__":
    
    print("====== START PREPROCESSING =======")
    
    # 1. Load config file
    config_data = helper.load_config()

    # 2. Load dataset
    train_set, valid_set, test_set = load_dataset(config_data)

    # 3. Check nan values
    # 3.1. Train data
    nan_detector(
        train_set,
        config_data
    )

    # 3.2. Valid data
    nan_detector(
        valid_set,
        config_data
    )

    # 3.3. Test data
    nan_detector(
        test_set,
        config_data
    )
    
    # TIDAK BISA DIPAKAI SAAT MODELING, KARENA ADA VALUE INF
    # 4. Transform outlier using log transform
    # 4.1 Train set
    
    X_train_feng = log_transform(
        train_set,
        config_data
    )

    # 4.2 Validation set

    X_valid_feng = log_transform(
        valid_set,
        config_data
    )

    # 4.3 Test set

    X_test_feng = log_transform(
        test_set,
        config_data
    )

    # 5. Save new data into pickle
    helper.dump_pickle(
        X_train_feng,
        config_data["train_feng_set_path"]
    )

    helper.dump_pickle(
        X_valid_feng,
        config_data["valid_feng_set_path"]
    )

    helper.dump_pickle(
        X_test_feng,
        config_data["test_feng_set_path"]
    )

    print("====== END PREPROCESSING =======")