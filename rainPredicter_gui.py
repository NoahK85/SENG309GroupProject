import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
from sklearn import tree

# Load normalized weather dataset
df = pd.read_csv("normalized_weather_data.csv")

X = df[["Temperature", "Humidity", "Wind Speed"]].values
y = df["Weather Type"].values

clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

# Try method for checking if pickle model exists
try:
    reg_model = joblib.load("weather_data_model.pkl")
except:
    reg_model = None


# Prediction function f
def make_prediction():
    try:
        t = float(temp_in.get())
        h = float(hum_in.get())
        w = float(wind_in.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers.")
        return

    # Normalize data using original model formulas
    scaled = [(t + 25) / 134, (h - 20) / 89, w / 48.5]

    # The weather type
    pred_type = clf.predict([scaled])
    type_result.config(text=f"Predicted Weather Type: {pred_type[0]}")

    # The precipitation %
    if reg_model:
        pred_precip = reg_model.predict([scaled])
        precip_result.config(
            text=f"Estimated Precipitation: {pred_precip[0] * 100:.2f}%"
        )
    else:
        precip_result.config(text="Estimated Precipitation: No model file found")


# Simple window layout for the user
win = tk.Tk()
win.title("Weather Predictor App")
win.geometry("320x380")

# User inputs
tk.Label(win, text="Temperature (°C):").pack(pady=5)
temp_in = tk.Entry(win)
temp_in.pack()

tk.Label(win, text="Humidity (%):").pack(pady=5)
hum_in = tk.Entry(win)
hum_in.pack()

tk.Label(win, text="Wind Speed (mph):").pack(pady=5)
wind_in = tk.Entry(win)
wind_in.pack()

# The action button
btn = tk.Button(win, text="Run Prediction", command=make_prediction)
btn.pack(pady=15)

# Final results
precip_result = tk.Label(win, text="Estimated Precipitation: ")
precip_result.pack(pady=5)

type_result = tk.Label(win, text="Predicted Weather Type: ")
type_result.pack(pady=5)

win.mainloop()
