import joblib
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix
)

# train decision tree
csvPd = pd.read_csv('normalized_weather_data.csv')

labels = []
for i in range(csvPd.get('Temperature').size):
    labels.append([csvPd.get('Temperature')[i],
                   csvPd.get('Humidity')[i], csvPd.get('Wind Speed')[i]])

# the y target variants for assignment evaluations
X = labels
y_weather_type = csvPd.get('Weather Type')
y_precipitation_pct = csvPd.get('Precipitation (%)') if 'Precipitation (%)' in csvPd else np.zeros(len(labels))

# Train the Decision Tree Classifier directly on the data
clf = tree.DecisionTreeClassifier(max_depth=5, random_state=42)
clf = clf.fit(X, y_weather_type)

clf_reg = tree.DecisionTreeRegressor(max_depth=5, random_state=42)
clf_reg = clf_reg.fit(X, y_precipitation_pct)

# Load our trained Linear Regression model
model = joblib.load('weather_data_model.pkl')

# get values from input
print("Welcome to the precipitation predictor")
print(" Please input temperature (degrees celsius): ", end="")
inputTemp = float(input())
print(" Please input humidity (%): ", end="")
inputHum = float(input())
print(" please input wind speed (mph): ", end="")
inputWind = float(input())
print("Predicting precipitation...")

# Define the weather vector
weather = [
    (inputTemp + 25) / 134,  # temp
    (inputHum - 20) / 89,  # humidity
    (inputWind) / 48.5,  # wind speed
]

weathers = [
    weather
]

# Linear Regression Predictions
lr_precip_prediction = model.predict(weathers)
predicted_value = lr_precip_prediction[0]

unique_classes = sorted(list(set(y_weather_type)))
lr_all_preds = model.predict(X)
lr_idx_preds = np.clip(np.round(lr_all_preds * (len(unique_classes) - 1)), 0, len(unique_classes) - 1).astype(int)
lr_str_preds = np.array([unique_classes[idx] for idx in lr_idx_preds])

# Map user input prediction to categorical text class for Linear Regression
user_lr_idx = np.clip(np.round(predicted_value * (len(unique_classes) - 1)), 0, len(unique_classes) - 1).astype(int)
lr_user_weather_type = unique_classes[user_lr_idx]

# Decision Tree Predictions
predicted_weather_arr = clf.predict([weather])
predicted_weather = str(predicted_weather_arr[0])
dt_precip_prediction = clf_reg.predict(weathers)[0]

print("\n" + "=" * 70)
print("             SYSTEM PERFORMANCE EVALUATION RESULTS")
print("=" * 70)

# Batch prediction metrics for entire dataset
dt_all_str_preds = clf.predict(X)
dt_all_precip_preds = clf_reg.predict(X)

class_to_idx = {name: i for i, name in enumerate(unique_classes)}
y_true_cat_num = np.array([class_to_idx[item] for item in y_weather_type])
dt_all_cat_num = np.array([class_to_idx[item] for item in dt_all_str_preds])

# Print Linear Regression Metrics
print("LINEAR REGRESSION PERFORMANCE RESULTS")
print(f" - Mean Absolute Error (MAE)      : {mean_absolute_error(y_precipitation_pct, lr_all_preds):.4f}")
print(f" - Mean Squared Error (MSE)       : {mean_squared_error(y_precipitation_pct, lr_all_preds):.4f}")
print(f" - Root Mean Squared Error (RMSE) : {root_mean_squared_error(y_precipitation_pct, lr_all_preds):.4f}")
print(f" - R-squared Performance Score (R2): {r2_score(y_precipitation_pct, lr_all_preds):.4f}")
print(f" - Mapped Classification Accuracy : {accuracy_score(y_weather_type, lr_str_preds) * 100:.2f}%")
print("\nLinear Regression Mapped Confusion Matrix:")
print(confusion_matrix(y_weather_type, lr_str_preds, labels=unique_classes))

# Print Decision Tree Metrics
print("\n" + "-" * 65)
print("DECISION TREE PERFORMANCE RESULTS")
print(f" - Continuous Prediction Error (MAE) : {mean_absolute_error(y_precipitation_pct, dt_all_precip_preds):.4f}")
print(f" - Continuous Prediction Error (MSE) : {mean_squared_error(y_precipitation_pct, dt_all_precip_preds):.4f}")
print(f" - Continuous Prediction Error (RMSE): {root_mean_squared_error(y_precipitation_pct, dt_all_precip_preds):.4f}")
print(f" - Continuous R2 Performance Score   : {r2_score(y_precipitation_pct, dt_all_precip_preds):.4f}")
print(f" - Overall Classification Accuracy   : {accuracy_score(y_weather_type, dt_all_str_preds) * 100:.2f}%")
print("\nDecision Tree Classifier Confusion Matrix:")
print(confusion_matrix(y_weather_type, dt_all_str_preds, labels=unique_classes))
print("=" * 70 + "\n")

# Results for both models
print("Weather details :")
print(f"- {-25 + weather[0] * 134:,.5f} temperature (Celsius)")
print(f"- {20 + weather[1] * 89:,.5f} humidity (%)")
print(f"- {weather[2] * 48.5:,.5f} wind speed (mph)")
print("-" * 45)
print("LINEAR REGRESSION MODEL OUTPUT")
print(f" -> Predicted Precipitation       : {predicted_value * 100:.2f}%")
print(f" -> Mapped Weather Type Condition : {lr_user_weather_type}")
print("\nDECISION TREE MODEL OUTPUT")
print(f" -> Predicted Precipitation       : {dt_precip_prediction * 100:.2f}%")
print(f" -> Predicted Weather Type Category: {predicted_weather}")
