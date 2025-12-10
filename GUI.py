import customtkinter as ctk

# Set theme and appearance
ctk.set_appearance_mode("light")          # "light" or "dark"
ctk.set_default_color_theme("dark-blue")      # "green", "dark-blue"

app = ctk.CTk()
app.title("New Zealand Crime Rate Prediction")
app.geometry("600x750")

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
        ctk.set_appearance_mode("dark")
    else:                             # If switch is OFF
        ctk.set_appearance_mode("light")

# --- Toggle Switch ---
mode_switch = ctk.CTkSwitch(
    app,
    text="Dark Mode",
    font=("Arial", 15),
    command=toggle_mode
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
    text="Which Year Do You Want To Predict?",
    font=("Arial", 13)
)
year_label.pack(pady=8)

# Dropdown for year/option selection
def option_changed(choice):
    output_box.delete("1.0", "end")
    output_box.insert("end", f"You selected: {choice}\n")
    output_box.insert("end", "Processing prediction...\n")

dropdown = ctk.CTkOptionMenu(
    year_frame,
    values=["2025", "2026", "2027", "2028", "2029", "2030"],
    command=option_changed,
    width=200
)
dropdown.pack(pady=8)

# Predict Button
def predict_crime():
    selected_year = dropdown.get()
    output_box.delete("1.0", "end")
    output_box.insert("end", f"Predicting crime rates for {selected_year}...\n")
    output_box.insert("end", "="*40 + "\n")
    output_box.insert("end", "Prediction results will appear here.\n")
    output_box.insert("end", "="*40 + "\n")

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

app.mainloop()