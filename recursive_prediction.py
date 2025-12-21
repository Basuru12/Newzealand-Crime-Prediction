"""
Recursive Prediction Script for Total Crime Values
Uses population.joblib and total.joblib models to predict total values
for a user-selected year using recursive forecasting.
"""

import joblib
import pandas as pd
import numpy as np


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

    Parameters
    ----------
    X_last : pd.DataFrame
        Last known row of features (t)
    population_model : trained model
        Model that predicts Population_lagged
    total_model : trained model
        Model that predicts total
    population_feature_name : str
        Column name of Population_lagged
    horizon : int
        Number of recursive steps

    Returns
    -------
    pd.DataFrame
        Forecasted Population and Total values with corresponding years
    """
    X_current = X_last.copy()
    forecasts = []

    for step in range(1, horizon + 1):
        # ---- Predict Population_lagged ----
        pop_pred = population_model.predict(X_current)[0]
        
        # Update population feature
        X_current[population_feature_name] = pop_pred

        # ---- Predict total ----
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


def load_models():
    """Load both population and total models"""
    try:
        population_model = joblib.load("population.joblib")
        total_model = joblib.load("total.joblib")
        print("✓ Models loaded successfully!")
        return population_model, total_model
    except FileNotFoundError as e:
        print(f"Error: Model file not found - {e}")
        return None, None
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None


def get_last_known_data(data_file="New_csv.csv"):
    """Get the last known data point from the dataset"""
    try:
        df = pd.read_csv(data_file)
        df = df.sort_values("Year").reset_index(drop=True)
        
        # Get the last row
        last_row = df.iloc[-1]
        last_year = int(last_row["Year"])
        
        # Extract features needed for prediction
        feature_cols = ["Total_Lagged", "Year", "Population_lagged"]
        X_last = df[feature_cols].iloc[-1:].copy()
        
        print(f"✓ Last known data point loaded (Year: {last_year})")
        return X_last, last_year
    except FileNotFoundError:
        print(f"Error: Dataset file '{data_file}' not found!")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def predict_total_for_year(target_year: int, data_file="New_csv.csv"):
    """
    Predict total crime value for a specific year using recursive forecasting.
    
    Parameters
    ----------
    target_year : int
        The year to predict total values for
    data_file : str
        Path to the dataset file containing lagged features
        
    Returns
    -------
    float or None
        Predicted total value for the target year, or None if error
    """
    # Load models
    population_model, total_model = load_models()
    if population_model is None or total_model is None:
        return None
    
    # Get last known data
    X_last, last_year = get_last_known_data(data_file)
    if X_last is None:
        return None
    
    # Validate target year
    if target_year <= last_year:
        print(f"Error: Target year ({target_year}) must be greater than last known year ({last_year})")
        return None
    
    # Calculate forecast horizon
    horizon = target_year - last_year
    
    print(f"\nForecasting from {last_year} to {target_year} (horizon: {horizon} steps)")
    print("-" * 50)
    
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
    
    # Display all intermediate predictions
    print("\nForecast Results:")
    print("=" * 50)
    for _, row in forecasts.iterrows():
        print(f"Year {int(row['Year'])}: Total = {row['total_pred']:,.0f}, "
              f"Population_lagged = {row['Population_lagged_pred']:,.0f}")
    print("=" * 50)
    
    return float(target_prediction)


def main():
    """Main function for interactive prediction"""
    print("=" * 60)
    print("New Zealand Crime Prediction - Recursive Model")
    print("=" * 60)
    
    # Get target year from user
    try:
        target_year = int(input("\nEnter the year to predict total values for: "))
        
        # Predict
        prediction = predict_total_for_year(target_year)
        
        if prediction is not None:
            print(f"\n✓ Prediction for Year {target_year}:")
            print(f"  Predicted Total: {prediction:,.0f}")
        else:
            print("\n✗ Prediction failed. Please check the error messages above.")
            
    except ValueError:
        print("Error: Please enter a valid year (integer)")
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()

