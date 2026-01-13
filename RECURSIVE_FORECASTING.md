# Recursive Forecasting for New Zealand Crime Prediction

## Overview

This system uses **two machine learning models jointly** to predict future crime rates in New Zealand using recursive forecasting.

## How It Works

### The Two Models

1. **`population.joblib`** (Lasso Regression)
   - Predicts `Population_lagged` for future years
   - Uses features: `Total_Lagged`, `Year`, `Population_lagged`

2. **`total.joblib`** (Gradient Boosting Regressor)
   - Predicts `Total` crime values
   - Uses features: `Total_Lagged`, `Year`, `Population_lagged`

### Recursive Forecasting Process

The forecasting is **recursive**, meaning each step uses predictions from the previous step:

```
Step 1:
  ┌─────────────────────────────────────┐
  │ Last Known Data (2024)              │
  │ - Total_Lagged: 216,533             │
  │ - Year: 2024                        │
  │ - Population_lagged: 5,223,100     │
  └─────────────────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────────┐
  │ population_model.predict()          │
  │ → Predicts Population for 2025     │
  └─────────────────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────────┐
  │ Update Population_lagged           │
  │ → Use predicted population          │
  └─────────────────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────────┐
  │ total_model.predict()               │
  │ → Predicts Total Crime for 2025    │
  │   (uses predicted population)       │
  └─────────────────────────────────────┘

Step 2:
  ┌─────────────────────────────────────┐
  │ Updated Data (2025)                │
  │ - Total_Lagged: [predicted from S1]  │
  │ - Year: 2025                        │
  │ - Population_lagged: [predicted S1] │
  └─────────────────────────────────────┘
           │
           ▼
  [Repeat process for 2026...]
```

### Key Features

1. **Joint Model Usage**: Both models work together - the population model feeds into the total model
2. **Recursive Updates**: Each step's predictions become inputs for the next step
3. **Multi-step Forecasting**: Can predict multiple years ahead by chaining predictions

### Example: Predicting 2027

If last known year is 2024 and target is 2027:

1. **Step 1** (2024 → 2025):
   - Predict Population for 2025 using 2024 data
   - Predict Total Crime for 2025 using predicted population

2. **Step 2** (2025 → 2026):
   - Use 2025 predictions as input
   - Predict Population for 2026
   - Predict Total Crime for 2026

3. **Step 3** (2026 → 2027):
   - Use 2026 predictions as input
   - Predict Population for 2027
   - Predict Total Crime for 2027 ← **Final Answer**

## Usage

### In Python Code

```python
import connection

# Predict total crime for a specific year
prediction = connection.predict_total_for_year_recursive(2027)
print(f"Predicted total crime for 2027: {prediction:,.0f}")

# Get full forecast with all intermediate steps
forecasts = connection.get_full_forecast(2027)
print(forecasts)
```

### In the UI

1. Select a target year (2025-2030)
2. Click "Predict Total (Recursive Model)"
3. View the complete recursive forecasting process:
   - Each step's population prediction
   - Each step's total crime prediction
   - How predictions feed into the next step
   - Final prediction for the target year

## Technical Details

### Features Used

- **Total_Lagged**: Previous year's total crime (lagged feature)
- **Year**: The year being predicted
- **Population_lagged**: Previous year's population (lagged feature)

### Why Recursive?

Recursive forecasting is necessary because:
- Future population depends on current population
- Future crime depends on current crime AND future population
- We need to predict both simultaneously, step by step

### Model Files

- `population.joblib`: Trained Lasso model for population prediction
- `total.joblib`: Trained Gradient Boosting model for total crime prediction
- `New_csv.csv`: Historical data with lagged features

## Advantages

1. **Realistic**: Models the actual dependencies between population and crime
2. **Accurate**: Uses the best available information at each step
3. **Transparent**: Shows all intermediate predictions
4. **Flexible**: Can forecast any number of years ahead

