import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np

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

# CALCULATION OF MODELS ACCURACY
unique_classes = sorted(list(set(y)))

# 1. Decision Tree Accuracy Computation
dt_all_preds = clf.predict(X)
dt_accuracy_percentage = accuracy_score(y, dt_all_preds) * 100

# 2. Linear Regression Accuracy Computation
if reg_model:
    lr_all_continuous_preds = reg_model.predict(X)
    lr_idx_preds = np.clip(np.round(lr_all_continuous_preds * (len(unique_classes) - 1)), 0,
                           len(unique_classes) - 1).astype(int)
    lr_all_categorical_preds = np.array([unique_classes[idx] for idx in lr_idx_preds])
    lr_accuracy_percentage = accuracy_score(y, lr_all_categorical_preds) * 100
else:
    lr_accuracy_percentage = 0.0


# Prediction function
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

    if model_choice.get() == "Linear Regression":
        # Linear Regression Prediction
        if reg_model:
            pred_precip = reg_model.predict([scaled])
            precip_result.config(text=f"Estimated Precipitation: {pred_precip[0] * 100:.2f}%")
            mapped_idx = int(max(0, min(len(unique_classes) - 1, round(pred_precip[0] * (len(unique_classes) - 1)))))
            type_result.config(text=f"Predicted Weather Type: {unique_classes[mapped_idx]}")

            accuracy_result.config(text=f"Model Testing Accuracy: {lr_accuracy_percentage:.2f}%", fg="#10ac84")
        else:
            precip_result.config(text="Estimated Precipitation: No model file found")
            type_result.config(text="Predicted Weather Type: No model file found")
            accuracy_result.config(text="Model Testing Accuracy: N/A (Model file missing)", fg="#e74c3c")
    else:
        # Decision Tree Prediction
        pred_type = clf.predict([scaled])
        type_result.config(
            text=f"Predicted Weather Type: {pred_type[0]}")

        try:
            class_idx = unique_classes.index(pred_type[0])
            mapped_precip = class_idx / max(1, (len(unique_classes) - 1))
            precip_result.config(text=f"Estimated Precipitation: {mapped_precip * 100:.2f}%")
        except ValueError:
            precip_result.config(text="Estimated Precipitation: Mapping error")

        accuracy_result.config(text=f"Model Testing Accuracy: {dt_accuracy_percentage:.2f}%", fg="#2e86de")


# Window layout for the user input
win = tk.Tk()
win.title("Weather Predictor App")
win.geometry("340x490")

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

tk.Label(win, text="Select Model Option:", font=("Arial", 9, "bold")).pack(pady=5)
model_choice = tk.StringVar(value="Linear Regression")

rb_lr = tk.Radiobutton(win, text="Linear Regression Model", variable=model_choice, value="Linear Regression")
rb_lr.pack(anchor="w", padx=70)

rb_dt = tk.Radiobutton(win, text="Decision Tree Model", variable=model_choice, value="Decision Tree")
rb_dt.pack(anchor="w", padx=70)

# The action button
btn = tk.Button(win, text="Run Prediction", command=make_prediction)
btn.pack(pady=15)

# Final results
precip_result = tk.Label(win, text="Estimated Precipitation: ")
precip_result.pack(pady=5)

type_result = tk.Label(win, text="Predicted Weather Type: ")
type_result.pack(pady=5)

# Accuracy Display Label
accuracy_result = tk.Label(win, text="Model Testing Accuracy: ..", font=("Arial", 10, "bold"))
accuracy_result.pack(pady=5)

win.mainloop()
