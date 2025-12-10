import customtkinter as ctk

# Set theme and appearance
ctk.set_appearance_mode("light")          # "light" or "dark"
ctk.set_default_color_theme("dark-blue")      # "green", "dark-blue"

app = ctk.CTk()
app.title("My App")
app.geometry("500x400")

label = ctk.CTkLabel(app, text="This is a Project to predict the future Crime rates in Newzeland",
                     font=("Arial", 12))
label.pack()

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
    command=toggle_mode
)
mode_switch.place(relx=1.0, rely=0.0, anchor="ne")

frame = ctk.CTkFrame(app)
frame.pack(padx=20, pady=20)
label = ctk.CTkLabel(frame, text="Let’s  Predict The Future")
label.pack()

# Add a button
button = ctk.CTkButton(app, text="Click Me")
button.pack(pady=20)

btn = ctk.CTkButton(app, text="Login", command=lambda: print("Clicked"),
                    fg_color="green", hover_color="darkgreen")
btn.pack()



app.mainloop()