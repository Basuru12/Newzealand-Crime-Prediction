import customtkinter as ctk
import connection

# Set theme and appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

app = ctk.CTk()
app.title("New Zealand Population Prediction")
app.geometry("600x700")

# Title Label
title_label = ctk.CTkLabel(
    app,
    text="New Zealand Population Prediction",
    font=("Arial", 18, "bold"),
    wraplength=500
)
title_label.pack(pady=(20, 10))

# Description Label
desc_label = ctk.CTkLabel(
    app,
    text="Predict population for year 2025 using population.joblib model",
    font=("Arial", 12),
    wraplength=500,
    justify="center"
)
desc_label.pack(pady=(0, 20))

# --- Toggle Function ---
def toggle_mode():
    if mode_switch.get() == 1:
        ctk.set_appearance_mode("light")
    else:
        ctk.set_appearance_mode("dark")

# --- Toggle Switch ---
mode_switch = ctk.CTkSwitch(
    app,
    text="Light Mode",
    font=("Arial", 14),
    command=toggle_mode,
)
mode_switch.place(relx=0.98, rely=0.02, anchor="ne")

# Prediction Frame
prediction_frame = ctk.CTkFrame(app)
prediction_frame.pack(padx=20, pady=(10, 10), fill="x")

prediction_title = ctk.CTkLabel(
    prediction_frame,
    text="Population Prediction for 2025",
    font=("Arial", 16, "bold")
)
prediction_title.pack(pady=15)

# Info Frame
info_frame = ctk.CTkFrame(app)
info_frame.pack(padx=20, pady=(5, 10), fill="x")

info_label = ctk.CTkLabel(
    info_frame,
    text="Model Information:",
    font=("Arial", 13, "bold")
)
info_label.pack(pady=(10, 5))

model_info = ctk.CTkLabel(
    info_frame,
    text="• Model: population.joblib (Lasso Regression)\n• Features: Total_Lagged, Year, Population_lagged\n• Data Source: data/New_csv.csv",
    font=("Arial", 11),
    justify="left"
)
model_info.pack(pady=(0, 10))

# Predict Button Function
def predict_population():
    output_box.delete("1.0", "end")
    
    try:
        # Check if model is loaded
        if not connection.is_model_loaded():
            output_box.insert("end", "Loading population model...\n")
            if not connection.load_population_model():
                output_box.insert("end", "ERROR: Model not loaded!\n")
                output_box.insert("end", "Please ensure 'Models/population.joblib' exists.\n")
                return
        
        # Get last known data
        try:
            X_last, last_year = connection.get_last_known_data()
            output_box.insert("end", f"Last known year: {last_year}\n")
            output_box.insert("end", f"Target year: 2025\n")
            output_box.insert("end", "-"*50 + "\n\n")
            
            # Display input features
            output_box.insert("end", "Input Features (from last known data):\n")
            output_box.insert("end", f"  • Total_Lagged: {X_last['Total_Lagged'].iloc[0]:,.0f}\n")
            output_box.insert("end", f"  • Year: {int(X_last['Year'].iloc[0])}\n")
            output_box.insert("end", f"  • Population_lagged: {X_last['Population_lagged'].iloc[0]:,.0f}\n")
            output_box.insert("end", "\n" + "-"*50 + "\n\n")
        except Exception as e:
            output_box.insert("end", f"Warning: Could not load data: {str(e)}\n")
            output_box.insert("end", "Attempting prediction anyway...\n\n")
        
        # Make prediction
        output_box.insert("end", "Making prediction...\n")
        prediction = connection.predict_population_for_year(2025)
        
        # Display results
        output_box.insert("end", "\n" + "="*50 + "\n")
        output_box.insert("end", "PREDICTION RESULT\n")
        output_box.insert("end", "="*50 + "\n")
        output_box.insert("end", f"\nYear: 2025\n")
        output_box.insert("end", f"Predicted Population: {prediction:,.0f}\n")
        output_box.insert("end", "="*50 + "\n")
        
        # Calculate change from last known
        try:
            X_last, last_year = connection.get_last_known_data()
            last_population = X_last['Population_lagged'].iloc[0]
            change = prediction - last_population
            change_percent = (change / last_population) * 100
            
            output_box.insert("end", f"\nChange from {last_year}:\n")
            output_box.insert("end", f"  • Absolute: {change:+,.0f}\n")
            output_box.insert("end", f"  • Percentage: {change_percent:+.2f}%\n")
        except:
            pass
        
    except ValueError as e:
        output_box.insert("end", f"ERROR: {str(e)}\n")
    except Exception as e:
        output_box.insert("end", f"ERROR during prediction: {str(e)}\n")
        output_box.insert("end", "Please check:\n")
        output_box.insert("end", "  • Model file exists: Models/population.joblib\n")
        output_box.insert("end", "  • Data file exists: data/New_csv.csv\n")

# Predict Button
predict_button = ctk.CTkButton(
    app,
    text="Predict Population for 2025",
    command=predict_population,
    width=250,
    height=40,
    font=("Arial", 14, "bold")
)
predict_button.pack(pady=15)

# Output Frame
output_frame = ctk.CTkFrame(app)
output_frame.pack(padx=20, pady=(10, 10), fill="both", expand=True)

output_label = ctk.CTkLabel(
    output_frame,
    text="Prediction Results:",
    font=("Arial", 13, "bold")
)
output_label.pack(pady=(10, 5))

# Create output textbox
output_box = ctk.CTkTextbox(output_frame, width=550, height=200)
output_box.pack(padx=10, pady=(0, 10), fill="both", expand=True)

# Initial message
output_box.insert("1.0", "Click 'Predict Population for 2025' to see results.\n\n")
output_box.insert("end", "The model will use the last known data from New_csv.csv\n")
output_box.insert("end", "to predict the population for year 2025.\n")

# Function to clear output
def clear_output():
    output_box.delete("1.0", "end")
    output_box.insert("end", "Output cleared.\n")

clear_button = ctk.CTkButton(
    app,
    text="Clear Output",
    command=clear_output,
    width=150
)
clear_button.pack(pady=(0, 10))

# Exit button function
def exit_app():
    app.quit()
    app.destroy()

# Exit button
exit_button = ctk.CTkButton(
    app,
    text="Exit",
    command=exit_app,
    width=80,
    fg_color="red",
    hover_color="darkred",
    font=("Arial", 12, "bold")
)
exit_button.place(relx=0.98, rely=0.99, anchor="se")

# Run the app
if __name__ == "__main__":
    app.mainloop()

