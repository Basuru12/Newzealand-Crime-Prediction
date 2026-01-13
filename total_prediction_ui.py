"""
Total Crime Prediction UI
=========================
A user interface for predicting total crime values using the trained model.
"""

import customtkinter as ctk
import joblib
import pandas as pd
import os
from pathlib import Path

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class TotalPredictionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("New Zealand Crime Prediction - Total Model")
        self.root.geometry("600x550")
        
        # Load the dataset
        self.df = None
        self.load_dataset()
        
        # Load the model
        self.model = None
        self.load_model()
        
        # Create UI elements
        self.create_widgets()
    
    def load_dataset(self):
        """Load the dataset from New_csv.csv."""
        data_path = Path("data/New_csv.csv")
        if not data_path.exists():
            data_path = Path("../data/New_csv.csv")
        
        if data_path.exists():
            try:
                self.df = pd.read_csv(data_path)
                self.df = self.df.sort_values("Year").reset_index(drop=True)
                print(f"Dataset loaded successfully from {data_path}")
            except Exception as e:
                print(f"Error loading dataset: {e}")
                self.df = None
        else:
            print(f"Dataset file not found at {data_path}")
            self.df = None
    
    def load_model(self):
        """Load the trained total model from the Models directory."""
        model_path = Path("Models/total.joblib")
        
        if not model_path.exists():
            # Try alternative path
            model_path = Path("../Models/total.joblib")
        
        if model_path.exists():
            try:
                self.model = joblib.load(model_path)
                print(f"Model loaded successfully from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
        else:
            print(f"Model file not found at {model_path}")
            self.model = None
    
    def create_widgets(self):
        """Create and arrange UI widgets."""
        # Title
        title_label = ctk.CTkLabel(
            self.root,
            text="Total Crime Prediction",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title_label.pack(pady=20)
        
        # Model status
        status_text = "Model Loaded ✓" if self.model is not None else "Model Not Found ✗"
        status_color = "green" if self.model is not None else "red"
        status_label = ctk.CTkLabel(
            self.root,
            text=status_text,
            font=ctk.CTkFont(size=14),
            text_color=status_color
        )
        status_label.pack(pady=5)
        
        # Input frame
        input_frame = ctk.CTkFrame(self.root)
        input_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        # Total_Lagged input
        self.total_lagged_label = ctk.CTkLabel(
            input_frame,
            text="Total_Lagged (Previous Year's Total Crime):",
            font=ctk.CTkFont(size=14)
        )
        self.total_lagged_label.pack(pady=(10, 5))
        
        self.total_lagged_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="e.g., 216533",
            width=400,
            height=35,
            font=ctk.CTkFont(size=14)
        )
        self.total_lagged_entry.pack(pady=5)
        
        # Year dropdown
        self.year_label = ctk.CTkLabel(
            input_frame,
            text="Select Year to Predict:",
            font=ctk.CTkFont(size=14)
        )
        self.year_label.pack(pady=(10, 5))
        
        # Get available years from dataset and add future years up to 2025
        year_options = []
        if self.df is not None:
            # Get years from dataset
            dataset_years = sorted(self.df["Year"].unique().tolist())
            year_options.extend([str(int(year)) for year in dataset_years])
            # Add future years up to 2025 only
            if dataset_years:
                last_year = int(dataset_years[-1])
                max_year = 2025
                for i in range(1, max_year - last_year + 1):
                    next_year = last_year + i
                    if next_year <= max_year:
                        year_options.append(str(next_year))
        else:
            # Fallback: generate years from 1980 to 2025
            year_options = [str(year) for year in range(1980, 2026)]
        
        self.year_dropdown = ctk.CTkComboBox(
            input_frame,
            values=year_options,
            width=400,
            height=35,
            font=ctk.CTkFont(size=14),
            command=self.on_year_selected
        )
        # Set default to last year + 1 (next year to predict)
        if year_options:
            if self.df is not None and len(self.df) > 0:
                last_year = int(self.df.iloc[-1]["Year"])
                next_year = str(last_year + 1)
                if next_year in year_options:
                    self.year_dropdown.set(next_year)
                else:
                    self.year_dropdown.set(year_options[-1])
            else:
                self.year_dropdown.set(year_options[-1])
        self.year_dropdown.pack(pady=5)
        
        # Population_lagged input
        self.population_lagged_label = ctk.CTkLabel(
            input_frame,
            text="Population_lagged (Previous Year's Population):",
            font=ctk.CTkFont(size=14)
        )
        self.population_lagged_label.pack(pady=(10, 5))
        
        self.population_lagged_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="e.g., 5223100",
            width=400,
            height=35,
            font=ctk.CTkFont(size=14)
        )
        self.population_lagged_entry.pack(pady=5)
        
        # Predict button
        self.predict_button = ctk.CTkButton(
            input_frame,
            text="Predict Total Crime",
            command=self.predict,
            width=400,
            height=40,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#1f538d",
            hover_color="#14375e"
        )
        self.predict_button.pack(pady=20)
        
        # Result frame
        result_frame = ctk.CTkFrame(self.root)
        result_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        self.result_label = ctk.CTkLabel(
            result_frame,
            text="Predicted Total Crime: ",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.result_label.pack(pady=10)
        
        self.result_value = ctk.CTkLabel(
            result_frame,
            text="--",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#4CAF50"
        )
        self.result_value.pack(pady=5)
        
        # Auto-load values for the default selected year after all widgets are created
        if self.df is not None:
            selected_year = self.year_dropdown.get()
            if selected_year:
                self.root.after(100, lambda: self.on_year_selected(selected_year))
    
    
    def on_year_selected(self, selected_year):
        """Automatically populate fields when a year is selected."""
        if self.df is None:
            return
        
        try:
            selected_year_int = int(selected_year)
            previous_year = selected_year_int - 1
            
            # Find the row for the previous year
            prev_year_row = self.df[self.df["Year"] == previous_year]
            
            if len(prev_year_row) > 0:
                # Previous year exists in dataset - use its Total and Population as lagged values
                row = prev_year_row.iloc[0]
                total_lagged = row["Total"]
                population_lagged = row["Population"]
            else:
                # Previous year not in dataset - use last known values
                last_row = self.df.iloc[-1]
                total_lagged = last_row["Total"]
                population_lagged = last_row["Population"]
            
            # Update the input fields
            self.total_lagged_entry.delete(0, "end")
            self.total_lagged_entry.insert(0, str(int(total_lagged)))
            
            self.population_lagged_entry.delete(0, "end")
            self.population_lagged_entry.insert(0, str(int(population_lagged)))
            
            # Auto-predict when year is selected
            self.predict()
            
        except Exception as e:
            print(f"Error in on_year_selected: {e}")
    
    def predict(self):
        """Make prediction using the loaded model."""
        if self.model is None:
            self.show_error("Model not loaded! Please ensure Models/total.joblib exists.")
            return
        
        try:
            # Get input values
            total_lagged = float(self.total_lagged_entry.get())
            year = float(self.year_dropdown.get())
            population_lagged = float(self.population_lagged_entry.get())
            
            # Create feature array in the correct order: ["Total_Lagged", "Year", "Population_lagged"]
            features = pd.DataFrame({
                "Total_Lagged": [total_lagged],
                "Year": [year],
                "Population_lagged": [population_lagged]
            })
            
            # Make prediction
            prediction = self.model.predict(features)
            
            # Handle array output (extract scalar if needed)
            if isinstance(prediction, (list, pd.Series, pd.Index)):
                prediction = prediction[0]
            elif hasattr(prediction, '__len__') and len(prediction) > 0:
                prediction = prediction[0]
            
            # Display result
            predicted_value = float(prediction)
            self.result_value.configure(text=f"{predicted_value:,.0f}")
            
        except ValueError as e:
            self.show_error("Please enter valid numeric values for all fields.")
        except Exception as e:
            self.show_error(f"Error making prediction: {e}")
    
    def show_error(self, message):
        """Display an error message."""
        error_window = ctk.CTkToplevel(self.root)
        error_window.title("Error")
        error_window.geometry("400x150")
        
        error_label = ctk.CTkLabel(
            error_window,
            text=message,
            font=ctk.CTkFont(size=14),
            text_color="red",
            wraplength=350
        )
        error_label.pack(pady=30)
        
        ok_button = ctk.CTkButton(
            error_window,
            text="OK",
            command=error_window.destroy,
            width=100
        )
        ok_button.pack(pady=10)


def main():
    """Main function to run the application."""
    root = ctk.CTk()
    app = TotalPredictionUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

