import customtkinter as ctk
import connection

# Set theme and appearance
ctk.set_appearance_mode("dark")          # "light" or "dark"
ctk.set_default_color_theme("dark-blue")      # "green", "dark-blue"

app = ctk.CTk()
app.title("New Zealand Crime Rate Prediction")
app.geometry("600x800")

label = ctk.CTkLabel(
    app,
    text="This is a Project to predict the future Crime rates in New Zealand",
    font=("Arial", 12),
    wraplength=450,  # Wrap text at 450 pixels
    justify="left"   # Align text to the left
)
label.pack(pady=(15, 10), padx=20, anchor="w")

# --- Toggle Function ---
def toggle_mode():
    if mode_switch.get() == 1:       # If switch is ON
        ctk.set_appearance_mode("light")
    else:                             # If switch is OFF
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
prediction_frame.pack(padx=20, pady=(5, 5), fill="x")

prediction_title = ctk.CTkLabel(
    prediction_frame,
    text="Let's Predict The Future",
    font=("Arial", 18, "bold")
)
prediction_title.pack(pady=8)

# Year Selection Frame
year_frame = ctk.CTkFrame(app)
year_frame.pack(padx=20, pady=(5, 5), fill="x")

year_label = ctk.CTkLabel(
    year_frame,
    text="Select Year or Enter Population:",
    font=("Arial", 13)
)
year_label.pack(pady=8)

# Dropdown for year/option selection
def option_changed(choice):
    # Auto-populate estimated population based on year
    year_to_pop = {
        "2025": "5310000",
        "2026": "5370000",
        "2027": "5430000",
        "2028": "5490000",
        "2029": "5550000",
        "2030": "5610000"
    }
    if choice in year_to_pop:
        population_entry.delete(0, "end")
        population_entry.insert(0, year_to_pop[choice])

dropdown = ctk.CTkOptionMenu(
    year_frame,
    values=["2025", "2026", "2027", "2028", "2029", "2030"],
    command=option_changed,
    width=200
)
dropdown.pack(pady=8)

# Population input
population_label = ctk.CTkLabel(
    year_frame,
    text="Population (for prediction):",
    font=("Arial", 12)
)
population_label.pack(pady=(8, 2))

population_entry = ctk.CTkEntry(
    year_frame,
    width=200,
    placeholder_text="e.g., 5310000"
)
population_entry.pack(pady=5)
population_entry.insert(0, "5310000")  # Default value

# Predict Button
def predict_crime():
    selected_year = dropdown.get()
    population_str = population_entry.get().strip()
    
    output_box.delete("1.0", "end")
    
    if not connection.is_model_loaded():
        output_box.insert("end", "ERROR: Model not loaded!\n")
        output_box.insert("end", "Please ensure 'lasso_alpha0.001_iter1000.joblib' exists.\n")
        return
    
    if not population_str:
        output_box.insert("end", "ERROR: Please enter a population value!\n")
        return
    
    try:
        # Convert population to numeric
        population = float(population_str)
        year = int(selected_year)
        
        # Make prediction using connection module
        prediction = connection.predict_crime_rate(population, year)
        
        # Display results
        output_box.insert("end", f"Year: {selected_year}\n")
        output_box.insert("end", f"Population: {population:,.0f}\n")
        output_box.insert("end", "="*40 + "\n")
        output_box.insert("end", f"Predicted Homicides: {prediction:,.0f}\n")
        output_box.insert("end", "="*40 + "\n")
        output_box.insert("end", "\n✓ Prediction based on Lasso Regression model\n")
        output_box.insert("end", f"Homicide rate per 100k: {(prediction/population*100000):.2f}\n")
        
    except ValueError:
        output_box.insert("end", "ERROR: Please enter a valid number for population!\n")
    except Exception as e:
        output_box.insert("end", f"ERROR during prediction: {str(e)}\n")
        output_box.insert("end", "Please check your input data.\n")

predict_button = ctk.CTkButton(
    app,
    text="Predict Crime Rate",
    command=predict_crime,
    width=200,
    height=35,
    font=("Arial", 13, "bold")
)
predict_button.pack(pady=10)

# Output Frame
output_frame = ctk.CTkFrame(app)
output_frame.pack(padx=20, pady=(5, 5), fill="both", expand=True)

output_label = ctk.CTkLabel(
    output_frame,
    text="Prediction Results:",
    font=("Arial", 13, "bold")
)
output_label.pack(pady=(8, 5))

# Create a textbox (the output box)
output_box = ctk.CTkTextbox(output_frame, width=500, height=150)
output_box.pack(padx=10, pady=(0, 8), fill="both", expand=True)

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

# Red exit button in bottom right corner
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

app.mainloop()