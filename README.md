# Student Performance Predictor

A complete Machine Learning web application to predict student performance based on various metrics.

## Project Structure

- **`backend/`**: Contains the web application code.
    - **`app.py`**: The Flask server.
    - **`templates/`**: Frontend HTML files.
- **`ml/`**: Machine Learning scripts.
    - **`train_model.py`**: Trains the model.
    - **`preprocess_data.py`**: Cleans data.
    - **`predict_performance.py`**: Inference script.
    - **`explore_data.py`**: Data exploration.
- **`models/`**: Stores the trained model (`student_performance_model.pkl`).
- **`data/`**: Directory for datasets.
- **`requirements.txt`**: Python dependencies.

## Setup & Running

### 1. Prerequisites
Ensure you have Python installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Usage

#### Run the Web Application
1. Navigate to the `backend` folder:
   ```bash
   cd backend
   ```
2. Start the server:
   ```bash
   python app.py
   ```
3. Open browser at: `http://127.0.0.1:5000/`

#### Train the Model (Optional)
1. Navigate to the `ml` folder:
   ```bash
   cd ml
   ```
2. Run the training script:
   ```bash
   python train_model.py
   ```
   *This saves the model to `../models/student_performance_model.pkl`*

#### Command Line Prediction
1. Navigate to the `ml` folder:
   ```bash
   cd ml
   ```
2. Run the prediction script:
   ```bash
   python predict_performance.py
   ```

## Features
- **Machine Learning**: Logistic Regression (`scikit-learn`).
- **Backend**: Flask.
- **Frontend**: Tailwind CSS + JS.
