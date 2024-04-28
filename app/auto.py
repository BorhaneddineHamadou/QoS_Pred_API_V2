from .auto_configure import *
from .auto_predict import *
import pathlib
from typing import Optional
from fastapi import FastAPI, Query, FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import pandas as pd
from pydantic import BaseModel
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import asyncio

RMSE_PATIENCE = 3
RMSE_THRESHOLD = 0.8
TRIALS = 3

auto_app = FastAPI()

BASE_DIR = pathlib.Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR.parent / "dataset"

# Global variables
processed_dataset = None
selected_columns = None
selected_horizon = None
hyperparameters = None
model = None
predictions = []
prediction_index = 0
rmse_table = []
is_task_running = False

async def find_and_retrain_model():
    global processed_dataset
    global selected_columns
    global selected_horizon
    global model
    global hyperparameters

    # Set is_task_running to True before starting the task
    global is_task_running
    is_task_running = True
    # Find and retrain the model asynchronously
    model, hyperparameters, training_elapsed_time = await find_best_model(processed_dataset, selected_columns, selected_horizon, TRIALS)

async def check_and_retrain_model():
    global rmse_table
    global is_task_running

    # Check if a task is already running
    if not is_task_running:
        # Check your condition to retrain the model
        if len(rmse_table) >= RMSE_PATIENCE and all(rmse > RMSE_THRESHOLD for rmse in rmse_table[-RMSE_PATIENCE:]):
            # Start find_and_retrain_model asynchronously
            await find_and_retrain_model()


@auto_app.post("/main")
async def main_function(
    file: UploadFile,
    target_columns: str = Form(...),  # Specify target columns as a comma-separated string
    forecasting_horizon: int = Form(...),  # Specify the forecasting horizon as an integer
):
    """ 
        Description: Upload and process a CSV or Excel file for time series forecasting.
        Parameters:
            file: The uploaded file (CSV or Excel).
            target_columns: a string specifying the target column (only one).
            forecasting_horizon: Integer specifying the forecasting horizon.
        Response: JSON containing the selected forecasting model and its hyperparameters.
    """
    global processed_dataset
    global selected_columns
    global selected_horizon

    # Check if the file format is CSV or Excel
    if file.filename.endswith((".csv", ".xlsx")):
        # Create the datasets directory if it doesn't exist
        os.makedirs(DATASET_DIR, exist_ok=True)

        # Save the uploaded file to the datasets directory
        dataset_path = os.path.join(DATASET_DIR, file.filename)
        with open(dataset_path, "wb") as f:
            f.write(file.file.read())

        # Read the dataset and identify real or integer columns
        try:
            df = pd.read_csv(dataset_path) if dataset_path.endswith(".csv") else pd.read_excel(dataset_path)
            processed_dataset = df.copy()  # Store the processed dataset
            numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
            
            # Parse the target columns from the comma-separated string
            target_columns = [col.strip() for col in target_columns.split(",")]

            if len(target_columns) > 1:
                return JSONResponse(content={"error": "Select only one column please."}, status_code=400)
        
            all_in_numerical = all(column in numeric_columns for column in target_columns)

            if all_in_numerical :
                selected_columns = target_columns
            else :
                return JSONResponse(content={"error": "Not all target columns are in the list of numerical columns.", "Numeric columns": numeric_columns}, status_code=400)
            
            if forecasting_horizon is not None and isinstance(forecasting_horizon, int) and forecasting_horizon > 0:
                selected_horizon = forecasting_horizon
            else :
                return JSONResponse(content={"error": "Forecasting horizon is not a valid integer or is not greater than 0."}, status_code=400)
            
            global hyperparameters
            global model

            model, hyperparameters, training_elapsed_time = find_best_model(processed_dataset, selected_columns, selected_horizon, TRIALS)
            
            # if best_params['forecasting_model'] in 

            return JSONResponse(content={"Best Params":hyperparameters}, status_code=200)
        except Exception as e:
            return JSONResponse(content={"error": f"Error processing the dataset: {str(e)}"}, status_code=400)
    else:
        return JSONResponse(content={"error": "Unsupported file format. Please upload a CSV or Excel file."}, status_code=400)

class InputData(BaseModel):
    input_data: str
@auto_app.get("/predict")
async def predict_endpoint(input_data: float):
    """
    Description: Make predictions for the next time steps based on the provided input data.
    Parameters:
        input_data: The actual value.
    Response: JSON containing the forecasted values.
    Note: The endpoint checks RMSE conditions and asynchronously re-trains the model if necessary.
    """
    global processed_dataset
    global selected_columns
    global selected_horizon
    global model
    global hyperparameters
    global predictions
    global prediction_index
    global rmse_table
    
    try:
        # Attempt to parse input_data as a float
        input_data = float(input_data)
    except ValueError:
        return HTTPException(status_code=400, detail="Please enter numerical value.")
    
    if not isinstance(input_data, (int, float)):
        return HTTPException(status_code=400, detail="Please enter valid value.")
    if processed_dataset is None or processed_dataset.empty :
        return HTTPException(status_code=400, detail="Dataset has not been uploaded and processed.")
    if selected_columns is None or len(selected_columns) == 0:
        return HTTPException(status_code=400, detail="You didn't select any columns yet.")
    if selected_horizon is None :
        return HTTPException(status_code=400, detail="You have to select a forecasting horizon.")
    
    new_row = {selected_columns[0]: input_data}
    processed_dataset = pd.concat([processed_dataset, pd.DataFrame([new_row])], ignore_index=True)

    in_data = processed_dataset[[selected_columns[0]]]

    if hyperparameters['forecasting_model'] in ["RNN", "LSTM", "GRU", "ESN"]:
        in_data = in_data[selected_columns[0]].tail(hyperparameters['look_back'])
    else :
        in_data = in_data[selected_columns[0]].tail(hyperparameters['window_length'])
    
    pred = auto_forecast(model, in_data.values, selected_horizon, hyperparameters)
    predictions.append(pred[0])

    # Calculate the RMSE
    if len(predictions) >= selected_horizon+1:
        rmse = np.sqrt(mean_squared_error(in_data.tail(selected_horizon), predictions[prediction_index]))
        prediction_index +=1
        rmse_table.append(rmse)

    # check_rmse_condition
    asyncio.create_task(check_and_retrain_model())

    return JSONResponse(content={f"prediction": str(pred[0])}, status_code=200)


@auto_app.put("/update_configs")
async def update_constants(new_rmse_patience: int, new_rmse_threshold: float, new_trials: int):
    """
    Description: Update constant variables (RMSE_PATIENCE, RMSE_THRESHOLD, TRIALS) for model re-training conditions.
    Parameters:
        new_rmse_patience: New value for RMSE patience.
        new_rmse_threshold: New value for RMSE threshold.
        new_trials: New value for the number of trials for optuna process.
    Response: JSON confirming the successful update and displaying the updated constant values.
    """
    global RMSE_PATIENCE, RMSE_THRESHOLD, TRIALS
    RMSE_PATIENCE = new_rmse_patience
    RMSE_THRESHOLD = new_rmse_threshold
    TRIALS = new_trials
    return {"message": "Constants updated successfully", "RMSE_PATIENCE": RMSE_PATIENCE, "RMSE_THRESHOLD": RMSE_THRESHOLD, "TRIALS": TRIALS}
