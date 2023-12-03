# Edge QoS Predictor API

<strong>QoS Predictor API</strong> is an API that utilizes advanced time series forecasting models to predict QoS values in edge environments. This API allows you to upload time series data, perform forecasting, and update model re-training conditions. Please follow the instructions below to use the API effectively.


## Getting Started

Follow these steps to launch the project on a Linux system. Please note that the commands may vary on Windows or macOS.

### Prerequisites

Make sure you have Python installed on your system.

### Installation

1. Optionally, you can create a virtual execution environment:

    ```bash
    python -m venv .
    ```

2. Activate the environment:

    ```bash
    source bin/activate
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

4. Launch the server:

    ```bash
    uvicorn app.auto:auto_app --reload
    ```
## Request the API
### Base URL

The base URL for the API is `http://127.0.0.1:8000`.

### 1. Upload and Process Time Series Data

#### `POST /main`

This endpoint allows you to upload and process a CSV or Excel file for time series forecasting.

**Parameters:**
- `file`: The CSV or Excel file containing time series data.
- `target_columns`: Specify the target column.
- `forecasting_horizon`: Specify the forecasting horizon as an integer.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/main" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/data.csv" \
  -F "target_columns=node1_delay" \
  -F "forecasting_horizon=7"
```
**Response:**
A JSON that includes the optimal model found for the series along with its parameters.
### 2. Make Predictions

#### `GET /predict`

This endpoint allows you to make predictions for the next time steps based on the provided input data.

**Parameters:**
- `input_data`: The actual input data (float).

**Example Request:**
```bash
curl -X GET "http://127.0.0.1:8000/predict?input_data=0.22"
```
**Response:**
A string-formatted list containing predictions (predicted values).
### 3. Update Model Re-training Conditions

#### `PUT /update_configs`

This endpoint allows you to update constant variables for model re-training conditions:

- `RMSE THRESHOLD`: This is a decimal number that sets the maximum allowed prediction error. If the error goes beyond this value, it's considered too high.
- `RMSE PATIENT`: This is an integer number. It represents how many times in a row we're okay with the error exceeding the threshold. If it happens more times consecutively, we might want to consider retraining the model.
- `TRIALS`: This is the number of times we repeat the process of finding the best model. A larger value might result in a better model, but it also takes more time.

**Parameters:**
- `new_rmse_patience`: New value for RMSE patience (int).
- `new_rmse_threshold`: New value for RMSE threshold (float).
- `new_trials`: New value for the number of trials (int).

**Example Request:**
```bash
curl -X PUT "http://127.0.0.1:8000/update_configs" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"new_rmse_patience": 5, "new_rmse_threshold": 0.75, "new_trials": 5}'
```

**Response:**

- Successful responses will include a JSON object with relevant information.
- Errors will be accompanied by an error message.

### Note

- Ensure that the file format is CSV or Excel for the `/main` endpoint.
- For the `/predict` endpoint, provide a valid float value for `input_data`.
- Use the `/update_configs` endpoint to adjust model re-training conditions.

Feel free to explore and interact with the API using the provided commands and examples. If you encounter any issues, refer to the error messages or consult the API documentation.


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

