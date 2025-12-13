import joblib
import numpy as np
import pandas as pd

# Load the trained model
model_data = None
model_loaded = False
feature_columns = None
training_data = None

def load_model():
    """Load the Lasso regression model from joblib file"""
    global model_data, model_loaded, feature_columns, training_data
    try:
        model_data = joblib.load('lasso_alpha0.001_iter1000.joblib')
        
        # Load training data to understand feature structure
        training_data = pd.read_csv('New_csv.csv')
        
        # Get feature columns (all except Year and the target)
        # The model was trained predicting Homicide, so exclude Year and Homicide
        feature_columns = [c for c in training_data.columns if c not in ['Year', 'Homicide']]
        
        model_loaded = True
        print(f"Model loaded successfully from lasso_alpha0.001_iter1000.joblib!")
        print(f"Model expects {len(feature_columns)} features: {feature_columns}")
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
        float: Predicted Homicide count based on population and average crime patterns
    """
    if not model_loaded:
        raise Exception("Model not loaded. Call load_model() first.")
    
    try:
        # Calculate the population ratio compared to most recent year
        recent_pop = training_data['Population'].iloc[-1]
        pop_ratio = population / recent_pop
        
        # Get the most recent row's features and scale them by population ratio
        recent_features = training_data[feature_columns].iloc[-1].copy()
        
        # Scale crime-related features by population ratio
        # Keep Population as the actual value
        for col in feature_columns:
            if col != 'Population' and col not in ['Crime_Index', 'Crime_Index_Lagged']:
                recent_features[col] = recent_features[col] * pop_ratio
        
        recent_features['Population'] = population
        
        # Prepare input array in correct order
        input_feature = np.array([recent_features.values])
        
        # Make prediction
        prediction = model_data.predict(input_feature)[0]
        return prediction
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")

# Auto-load model when module is imported
load_model()