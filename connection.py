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
        model_data = joblib.load('lasso_alpha_0_001_maxiter_1000.joblib')
        
        # Load training data to understand feature structure
        training_data = pd.read_csv('New_csv.csv')
        
        # Get feature columns (all except the target)
        # The model was trained predicting Homicide, so exclude only Homicide
        # Include Year as a feature
        feature_columns = [c for c in training_data.columns if c not in ['Homicide']]
        
        model_loaded = True
        print(f"Model loaded successfully from lasso_alpha_0_001_maxiter_1000.joblib!")
        print(f"Model expects {len(feature_columns)} features: {feature_columns}")
        return True
    except Exception as e:
        model_loaded = False
        print(f"Error loading model: {e}")
        return False

def is_model_loaded():
    """Check if model is loaded"""
    return model_loaded

def predict_crime_rate(population, year=None):
    """Make crime prediction based on population
    
    Args:
        population (float): Population value for prediction
        year (int, optional): Year for prediction. If None, uses next year after training data
        
    Returns:
        float: Predicted Homicide count based on population and average crime patterns
    """
    if not model_loaded:
        raise Exception("Model not loaded. Call load_model() first.")
    
    try:
        # Use average of last 3 years for more stable baseline
        recent_data = training_data.tail(3)
        avg_features = recent_data[feature_columns].mean()
        
        # Calculate the population ratio compared to average recent population
        avg_recent_pop = recent_data['Population'].mean()
        pop_ratio = population / avg_recent_pop
        
        # Set the year
        if year is not None:
            avg_features['Year'] = year
        else:
            avg_features['Year'] = training_data['Year'].iloc[-1] + 1
        
        # Scale crime-related features by population ratio
        # Keep Population as the actual value, and Year as specified
        for col in feature_columns:
            if col not in ['Population', 'Year', 'Crime_Index', 'Crime_Index_Lagged']:
                avg_features[col] = avg_features[col] * pop_ratio
        
        avg_features['Population'] = population
        
        # Update Crime_Index based on population ratio if it exists
        if 'Crime_Index' in feature_columns:
            avg_features['Crime_Index'] = avg_features['Crime_Index'] * pop_ratio
        if 'Crime_Index_Lagged' in feature_columns:
            avg_features['Crime_Index_Lagged'] = avg_features['Crime_Index_Lagged'] * pop_ratio
        
        # Prepare input array in correct order
        input_feature = np.array([avg_features.values])
        
        # Make prediction
        prediction = model_data.predict(input_feature)[0]
        
        # Ensure prediction is not negative (homicides can't be negative)
        # Also apply reasonable bounds
        prediction = max(0, prediction)
        
        # If prediction still seems unreasonable, use simple rate-based approach as fallback
        if prediction < 1:
            # Calculate average homicide rate from historical data
            avg_homicide_rate = (training_data['Homicide'] / training_data['Population']).mean()
            prediction = max(prediction, avg_homicide_rate * population)
        
        return prediction
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")

# Auto-load model when module is imported
load_model()