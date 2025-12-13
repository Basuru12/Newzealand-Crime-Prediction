import joblib
import numpy as np

# Load the trained model
model_data = None
model_loaded = False

def load_model():
    """Load the Lasso regression model from joblib file"""
    global model_data, model_loaded
    try:
        model_data = joblib.load('lasso_alpha0.001_iter1000.joblib')
        model_loaded = True
        print("Model loaded successfully from lasso_alpha0.001_iter1000.joblib!")
        return True
    except Exception as e:
        model_loaded = False
        print(f"Error loading model: {e}")
        return False

def is_model_loaded():
    """Check if model is loaded"""
    return model_loaded

def predict_crime_rate(population):
    """Make crime prediction based on population
    
    Args:
        population (float): Population value for prediction
        
    Returns:
        float: Predicted total crimes, or None if prediction fails
    """
    if not model_loaded:
        raise Exception("Model not loaded. Call load_model() first.")
    
    try:
        # Prepare input as 2D array (population as single feature)
        input_feature = np.array([[population]])
        
        # Make prediction
        prediction = model_data.predict(input_feature)[0]
        return prediction
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")

# Auto-load model when module is imported
load_model()