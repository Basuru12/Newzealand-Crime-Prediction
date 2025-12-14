import joblib
import numpy as np
import pandas as pd

# Load the trained model
model_data = None
model_loaded = False
feature_columns = None

def load_model():
    """Load the Lasso regression model from joblib file"""
    global model_data, model_loaded, feature_columns
    try:
        model_data = joblib.load('RandomForest_last.joblib')
        
        feature_columns = ["Year","Population"]
        
        model_loaded = True
        print(f"Model loaded successfully from RandomForest_last.joblib!")
        print(f"Model expects {len(feature_columns)} features: {feature_columns}")
        return True
    except Exception as e:
        model_loaded = False
        print(f"Error loading model: {e}")
        return False

def is_model_loaded():
    """Check if model is loaded"""
    return model_loaded

def predict_crime_rate(year=None, population=0):
    """Make crime prediction based on population
    
    Args:
        year (int, optional): Year for prediction
        population (float): Population value for prediction
        
    Returns:
        float: Predicted total crimes based on year and population
    """
    if not model_loaded:
        raise Exception("Model not loaded. Call load_model() first.")
    
    try:
        prediction = model_data.predict([[year, population]])
        
        # Return single value instead of array
        return prediction[0]
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")

# Auto-load model when module is imported
load_model()