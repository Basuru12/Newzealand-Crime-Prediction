import joblib
import numpy as np
import pandas as pd

# Load the trained model
model_data = None
model_loaded = False
feature_columns = None

# Recursive model variables
population_model = None
total_model = None
recursive_models_loaded = False

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


# ========== Recursive Model Functions ==========

def load_recursive_models():
    """Load both population and total models for recursive forecasting"""
    global population_model, total_model, recursive_models_loaded
    try:
        population_model = joblib.load("population.joblib")
        total_model = joblib.load("total.joblib")
        recursive_models_loaded = True
        return True
    except FileNotFoundError as e:
        recursive_models_loaded = False
        print(f"Error: Model file not found - {e}")
        return False
    except Exception as e:
        recursive_models_loaded = False
        print(f"Error loading recursive models: {e}")
        return False

def is_recursive_models_loaded():
    """Check if recursive models are loaded"""
    return recursive_models_loaded

def get_last_known_data(data_file="New_csv.csv"):
    """Get the last known data point from the dataset"""
    try:
        df = pd.read_csv(data_file)
        df = df.sort_values("Year").reset_index(drop=True)
        
        # Extract features needed for prediction
        feature_cols = ["Total_Lagged", "Year", "Population_lagged"]
        X_last = df[feature_cols].iloc[-1:].copy()
        last_year = int(df.iloc[-1]["Year"])
        
        return X_last, last_year
    except FileNotFoundError:
        raise Exception(f"Dataset file '{data_file}' not found!")
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def recursive_forecast_with_exogenous(
    X_last: pd.DataFrame,
    population_model,
    total_model,
    population_feature_name: str,
    horizon: int = 3
):
    """
    Recursive forecast where Population_lagged is predicted first
    and then used to predict total.
    """
    X_current = X_last.copy()
    forecasts = []

    for step in range(1, horizon + 1):
        # Predict Population_lagged
        pop_pred = population_model.predict(X_current)[0]
        
        # Update population feature
        X_current[population_feature_name] = pop_pred

        # Predict total
        total_pred = total_model.predict(X_current)[0]

        # Store predictions
        current_year = int(X_current["Year"].iloc[0])
        forecasts.append({
            "Year": current_year,
            "step": step,
            "Population_lagged_pred": pop_pred,
            "total_pred": total_pred
        })

        # Update features for next iteration
        X_current[population_feature_name] = pop_pred
        X_current["Total_Lagged"] = total_pred
        X_current["Year"] = current_year + 1

    return pd.DataFrame(forecasts)

def predict_total_for_year_recursive(target_year: int, data_file="New_csv.csv"):
    """
    Predict total crime value for a specific year using recursive forecasting.
    
    Args:
        target_year (int): The year to predict total values for
        data_file (str): Path to the dataset file containing lagged features
        
    Returns:
        float: Predicted total value for the target year
    """
    if not recursive_models_loaded:
        if not load_recursive_models():
            raise Exception("Failed to load recursive models. Ensure population.joblib and total.joblib exist.")
    
    # Get last known data
    X_last, last_year = get_last_known_data(data_file)
    
    # Validate target year
    if target_year <= last_year:
        raise Exception(f"Target year ({target_year}) must be greater than last known year ({last_year})")
    
    # Calculate forecast horizon
    horizon = target_year - last_year
    
    # Perform recursive forecast
    forecasts = recursive_forecast_with_exogenous(
        X_last=X_last,
        population_model=population_model,
        total_model=total_model,
        population_feature_name="Population_lagged",
        horizon=horizon
    )
    
    # Get the prediction for the target year
    target_prediction = forecasts[forecasts["Year"] == target_year]["total_pred"].iloc[0]
    
    return float(target_prediction)

# Auto-load models when module is imported
load_model()
load_recursive_models()