from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pandas as pd
import helper
import data_pipeline

config_data = helper.load_config()
model_data = helper.load_pickle(config_data["production_model_path"])

class Features(BaseModel):
    NA_Sales: float
    EU_Sales: float
    JP_Sales: float
    Other_Sales: float

app = FastAPI()

@app.get("/")
def home():
    return "FastAPI is up!"

@app.post("/predict/")
def predict(data: Features):
    # Convert data API into DataFrame
    data = pd.DataFrame([data]).reset_index(drop = True)

    # Check range data
    try:
        data_pipeline.check_input_data(data, config_data)
    
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}
    
    # Predict data
    y_pred = model_data["model_data"]["model_object"].predict(data)

    # if not y_pred:
    #     raise HTTPException(status_code=400, detail = "Model not found")

    return {"res": y_pred, "error_msg": ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)