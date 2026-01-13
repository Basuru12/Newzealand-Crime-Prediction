import joblib
import numpy as np
import pandas as pd
import os

# Model variables
population_model = None
model_loaded = False

def load_population_model():
    """Load the population prediction model from joblib file"""
    global population_model, model_loaded
    try:
        model_path = "Models/population.joblib"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        population_model = joblib.load(model_path)
        model_loaded = True
        print(f"Population model loaded successfully from {model_path}!")
        return True
    except Exception as e:
        model_loaded = False
        print(f"Error loading population model: {e}")
        return False

def is_model_loaded():
    """Check if model is loaded"""
    return model_loaded

def get_last_known_data(data_file="data/New_csv.csv"):
    """Get the last known data point from the dataset"""
    try:
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        df = pd.read_csv(data_file)
        df = df.sort_values("Year").reset_index(drop=True)
        
        # Extract features needed for prediction
        feature_cols = ["Total_Lagged", "Year", "Population_lagged"]
        
        # Check if all required columns exist
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        X_last = df[feature_cols].iloc[-1:].copy()
        last_year = int(df.iloc[-1]["Year"])
        
        return X_last, last_year
    except FileNotFoundError as e:
        raise Exception(f"Dataset file not found: {e}")
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def predict_population_for_year(target_year: int, data_file="data/New_csv.csv"):
    """
    Predict population for a specific year using the population model.
    
    Args:
        target_year (int): The year to predict population for (e.g., 2025)
        data_file (str): Path to the dataset file containing lagged features
        
    Returns:
        float: Predicted population value for the target year
    """
    if not model_loaded:
        if not load_population_model():
            raise Exception("Failed to load population model. Ensure Models/population.joblib exists.")
    
    # Get last known data
    X_last, last_year = get_last_known_data(data_file)
    
    # Validate target year
    if target_year <= last_year:
        raise Exception(f"Target year ({target_year}) must be greater than last known year ({last_year})")
    
    # Calculate forecast horizon
    horizon = target_year - last_year
    
    # Prepare input data for prediction
    # Ensure columns are in the correct order: Total_Lagged, Year, Population_lagged
    X_current = X_last[["Total_Lagged", "Year", "Population_lagged"]].copy()
    print(X_current)
    # Update the year to target year
    X_current.loc[X_current.index[0], "Year"] = target_year
    
    # Make prediction
    try:
        prediction = population_model.predict(X_current)
        
        # Extract scalar value from array
        if isinstance(prediction, np.ndarray):
            return float(prediction[0])
        return float(prediction)
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")

# Auto-load model when module is imported
load_population_model()

