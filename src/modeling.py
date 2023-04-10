from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

import json
from tqdm import tqdm
import pandas as pd
import copy 
import hashlib

import helper

def load_train_feng(params: dict) -> pd.DataFrame:
    # Load train set
    X_train = helper.load_pickle(params["train_feng_set_path"])
    y_train = helper.load_pickle(params["train_set_path"][1])

    return X_train, y_train

def load_valid_feng(params: dict) -> pd.DataFrame:
    # Load valid set
    X_valid = helper.load_pickle(params["valid_feng_set_path"])
    y_valid = helper.load_pickle(params["valid_set_path"][1])

    return X_valid, y_valid

def load_test_feng(params: dict) -> pd.DataFrame:
    # Load test set
    X_test = helper.load_pickle(params["test_feng_set_path"])
    y_test = helper.load_pickle(params["test_set_path"][1])

    return X_test, y_test

def load_dataset(params: dict) -> pd.DataFrame:
    # debug message
    helper.print_debug("Loading dataset...")

    # Load train set
    X_train, y_train = load_train_feng(params)

    # Load valid set
    X_valid, y_valid = load_valid_feng(params)

    # Load test set
    X_test, y_test = load_test_feng(params)

    # debug message
    helper.print_debug("Dataset loaded.")

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def training_log_template() -> dict:
    # Debug message
    helper.print_debug("Creating training log template")

    # Template of training log
    training_log = {
        "model_name" : [],
        "model_uid" : [],
        "training_time" : [],
        "training_date" : [],
        "r2_score" : [],
        "mse_score": [],
        "mae_score": [],
        "data_configurations" : [],
    }

    # Debug message
    helper.print_debug("Training log template created")

    return training_log

def training_log_updater(current_log: dict, params: dict) -> list:
    # Create copy of current log
    current_log = current_log.copy()

    # Path for training log file
    log_path = params["training_log_path"]

    try:
        with open(log_path, "r") as file:
            last_log = json.load(file)
        file.close()
    except FileNotFoundError as ffe:
        with open(log_path, "w") as file:
            file.write("[]")
        file.close()
        with open(log_path, "r") as file:
            last_log = json.load(file)
        file.close()
    
    last_log.append(current_log)

    with open(log_path, "w") as file:
        json.dump(last_log, file)
        file.close()

    return last_log

def create_model_object(params: dict) -> list:
    # Debug message
    helper.print_debug("Creating model objects")

    # Create model objects
    lr = LinearRegression()
    dct = DecisionTreeRegressor()
    rfr = RandomForestRegressor()
    knn = KNeighborsRegressor()

    # Create list of model
    list_of_model = {
        "vanilla" : [
        { "model_name": lr.__class__.__name__, "model_object": lr, "model_uid": ""},
        { "model_name": dct.__class__.__name__, "model_object": dct, "model_uid": ""},
        { "model_name": rfr.__class__.__name__, "model_object": rfr, "model_uid": ""},
        { "model_name": knn.__class__.__name__, "model_object": knn, "model_uid": ""},
        ]
    }

    # Debug message
    helper.print_debug("Model objects created")

    return list_of_model

def train_eval(configuration_model: str, params: dict, hyperparams_model: list = None):
    # Load dataset
    X_train, y_train, \
    X_valid, y_valid, \
    X_test, y_test = load_dataset(params)

    # Variable to store trained models
    list_of_trained_model = dict()

    # Create log template
    training_log = training_log_template()

    # Training for every data configuration
    for config_data in X_train:
        # Debug message
        helper.print_debug(f"Training model based on configuration data: {config_data}")

        # Create model objects
        if hyperparams_model == None:
            list_of_model = create_model_object(params)
        else:
            list_of_model = copy.deepcopy(hyperparams_model)

        # Variabel to store tained model
        trained_model = list()

        # Load train data based on its configuration
        X_train_data = X_train[config_data]
        y_train_data = y_train[config_data]

        # Train each model by current dataset configuration
        for model in list_of_model:
            # Debug message
            helper.print_debug("Training model: {}".format(model["model_name"]))

            # Training
            training_time = helper.timestamp()
            model["model_object"].fit(X_train_data, y_train_data)
            training_time = (helper.timestamp() - training_time).total_seconds()

            # Debug message
            helper.print_debug("Evalutaing model: {}".format(model["model_name"]))

            # Evaluation
            y_pred = model["model_object"].predict(X_valid)
            performance_r2 = r2_score(y_valid, y_pred)
            performance_mse = mean_squared_error(y_valid, y_pred)
            performance_mae = mean_absolute_error(y_valid, y_pred)


            # Debug message
            helper.print_debug("Logging: {}".format(model["model_name"]))

            # Create UID
            uid = hashlib.md5(str(training_time).encode()).hexdigest()

            # Assign model's UID
            model["model_uid"] = uid

            # Create training log data
            training_log["model_name"].append("{}-{}".format(configuration_model, model["model_name"]))
            training_log["model_uid"].append(uid)
            training_log["training_time"].append(training_time)
            training_log["training_date"].append(helper.timestamp())
            training_log["r2_score"].append(performance_r2)
            training_log["mse_score"].append(performance_mse)
            training_log["mae_score"].append(performance_mae)
            training_log["data_configurations"].append(config_data)

            # Collect current trained model
            trained_model.append(copy.deepcopy(model))

            # Debug message
            helper.print_debug("Model {} has been trained for configuration data {}.".format(model["model_name"], config_data))
        
        # Collect current trained list of model
        list_of_trained_model[config_data] = copy.deepcopy(trained_model)
    
    # Debug message
    helper.print_debug("All combination models and configuration data has been trained.")
    
    # Return list trained model
    return list_of_trained_model, training_log

def get_production_model(list_of_model, training_log, params):
    # Create copy list of model
    list_of_model = copy.deepcopy(list_of_model)
    
    # Debug message
    helper.print_debug("Choosing model by metrics score.")

    # Create required predefined variabel
    curr_production_model = None
    prev_production_model = None
    production_model_log = None

    # Debug message
    helper.print_debug("Converting training log type of data from dict to dataframe.")

    # Convert dictionary to pandas for easy operation
    training_log = pd.DataFrame(copy.deepcopy(training_log))

    # Debug message
    helper.print_debug("Trying to load previous production model.")

    # Check if there is a previous production model
    try:
        prev_production_model = helper.load_pickle(params["production_model_path"])
        helper.print_debug("Previous production model loaded.")

    except FileNotFoundError:
        helper.print_debug("No previous production model detected, choosing best model only from current trained model.")

    # If previous production model detected:
    if prev_production_model != None:
        # Debug message
        helper.print_debug("Loading validation data.")
        X_valid, y_valid = load_valid_feng(params)
        
        # Debug message
        helper.print_debug("Checking compatibilty previous production model's input with current train data's features.")

        # Check list features of previous production model and current dataset
        production_model_features = set(prev_production_model["model_data"]["model_object"].feature_names_in_)
        current_dataset_features = set(X_valid.columns)
        number_of_different_features = len((production_model_features - current_dataset_features) | (current_dataset_features - production_model_features))

        # If feature matched:
        if number_of_different_features == 0:
            # Debug message
            helper.print_debug("Features compatible.")

            # Debug message
            helper.print_debug("Reassesing previous model performance using current validation data.")

            # Re-predict previous production model to provide valid metrics compared to other current models
            y_pred = prev_production_model["model_data"]["model_object"].predict(X_valid)

            # Re-asses prediction result
            # TODO: fix this
            eval_res = classification_report(y_valid, y_pred, output_dict = True)

            # Debug message
            helper.print_debug("Assessing complete.")

            # Debug message
            helper.print_debug("Storing new metrics data to previous model structure.")

            # Update their performance log
            prev_production_model["model_log"]["performance"] = eval_res
            prev_production_model["model_log"]["f1_score_avg"] = eval_res["macro avg"]["f1-score"]

            # Debug message
            helper.print_debug("Adding previous model data to current training log and list of model")

            # Added previous production model log to current logs to compere who has the greatest f1 score
            training_log = pd.concat([training_log, pd.DataFrame([prev_production_model["model_log"]])])

            # Added previous production model to current list of models to choose from if it has the greatest f1 score
            list_of_model["prev_production_model"] = [copy.deepcopy(prev_production_model["model_data"])]
        else:
            # To indicate that we are not using previous production model
            prev_production_model = None

            # Debug message
            helper.print_debug("Different features between production model with current dataset is detected, ignoring production dataset.")

    # Debug message
    helper.print_debug("Sorting training log by f1 macro avg and training time.")

    # Sort training log by f1 score macro avg and trining time
    best_model_log = training_log.sort_values(["r2_score", "training_time"], ascending = [False, True]).iloc[0]
    
    # Debug message
    helper.print_debug("Searching model data based on sorted training log.")

    # Get model object with greatest f1 score macro avg by using UID
    for configuration_data in list_of_model:
        for model_data in list_of_model[configuration_data]:
            if model_data["model_uid"] == best_model_log["model_uid"]:
                curr_production_model = dict()
                curr_production_model["model_data"] = copy.deepcopy(model_data)
                curr_production_model["model_log"] = copy.deepcopy(best_model_log.to_dict())
                curr_production_model["model_log"]["model_name"] = "Production-{}".format(curr_production_model["model_data"]["model_name"])
                curr_production_model["model_log"]["training_date"] = str(curr_production_model["model_log"]["training_date"])
                production_model_log = training_log_updater(curr_production_model["model_log"], params)
                break
    
    # In case UID not found
    if curr_production_model == None:
        raise RuntimeError("The best model not found in your list of model.")
    
    # Debug message
    helper.print_debug("Model chosen.")

    # Dump chosen production model
    helper.dump_pickle(curr_production_model, params["production_model_path"])
    
    # Return current chosen production model, log of production models and current training log
    return curr_production_model, production_model_log, training_log


def create_dist_params(model_name: str) -> dict:
    # Define models parameters
    dist_params_lr = {}
    dist_params_dct = {
        "criterion" : ["squared_error", "absolute_error", "friedman_mse"],
        "min_samples_split" : [2, 4, 6, 10, 15, 20, 25],
        "min_samples_leaf" : [2, 4, 6, 10, 15, 20, 25]
    }
    dist_params_rfr = {
        "criterion" : ["squared_error", "absolute_error", "friedman_mse"],
        "n_estimators" : [50, 100, 200, 300, 400, 500],
        "min_samples_split" : [2, 4, 6, 10, 15, 20, 25],
        "min_samples_leaf" : [2, 4, 6, 10, 15, 20, 25]
    }
    dist_params_knn = {
        "algorithm" : ["ball_tree", "kd_tree", "brute"],
        "n_neighbors" : [2, 3, 4, 5, 6, 10, 15, 20, 25],
        "leaf_size" : [2, 3, 4, 5, 6, 10, 15, 20, 25],
    }

    # Make all models parameters into one dict
    dist_params = {
        "LinearRegression": dist_params_lr,
        "DecisionTreeRegressor": dist_params_dct,
        "RandomForestRegressor": dist_params_rfr,
        "KNeighborsRegressor": dist_params_knn
    }

    return dist_params[model_name]

def hyper_params_tuning(model: dict) -> list:
    # Create copy of current best baseline model
    model = copy.deepcopy(model)

    # Create model's parameter distribution
    dist_params = create_dist_params(model["model_data"]["model_name"])

    # Create model object
    model_rsc = RandomizedSearchCV(model["model_data"]["model_object"], dist_params, n_jobs = -1)
    model_data = {
        "model_name": model["model_data"]["model_name"],
        "model_object": model_rsc,
        "model_uid": ""
    }
    
    # Return model object
    return [model_data]