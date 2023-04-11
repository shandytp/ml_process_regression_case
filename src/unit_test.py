import pandas as pd
import numpy as np
import helper

config_data = helper.load_config()

def test_nan_detector():
    # Arrange
    mock_data = {"NA_Sales": [12.0, 39.0],
                 "JP_Sales": [2.5, 21.4],
                 "EU_Sales": [42.5, 21.9]}
    
    mock_data = pd.DataFrame(mock_data)
    # Act

    # create temp list for save .isna() value
    tmp_res = []

    # looping for all cols
    for col in mock_data.columns:
        count = mock_data[col].isnull().sum()
        tmp_res.append(count)

    # Assert
    assert sum(tmp_res) == 0

def test_shape_train_set():
    # Arrange
    X_train = helper.load_pickle(config_data["train_set_path"][0])
    y_train = helper.load_pickle(config_data["train_set_path"][1])

    # Act
    X_train_shape = X_train.shape
    y_train_shape = y_train.shape
    
    # Assert
    assert X_train_shape == (13278, 4)
    assert y_train_shape == (13278,)

def test_shape_valid_set():
    # Arrange
    X_valid = helper.load_pickle(config_data["valid_set_path"][0])
    y_valid = helper.load_pickle(config_data["valid_set_path"][1])

    # Act
    X_valid_shape = X_valid.shape
    y_valid_shape = y_valid.shape
    
    # Assert
    assert X_valid_shape == (1660, 4)
    assert y_valid_shape == (1660,)

def test_shape_test_set():
    # Arrange
    X_test = helper.load_pickle(config_data["test_set_path"][0])
    y_test = helper.load_pickle(config_data["test_set_path"][1])

    # Act
    X_test_shape = X_test.shape
    y_test_shape = y_test.shape
    
    # Assert
    assert X_test_shape == (1660, 4)
    assert y_test_shape == (1660,)
