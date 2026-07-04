import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)
import joblib

# Load our data set
df = pd.read_csv("normalized_weather_data.csv")

# Create the X and y arrays
X = df[["Temperature", "Humidity", "Wind Speed"]]
y = df["Precipitation (%)"]
y_categorical = df["Weather Type"]

X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = train_test_split(
    X, y, y_categorical, test_size=0.25, random_state=42
)

# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a file so we can use it to make predictions later
joblib.dump(model, 'weather_data_model.pkl')

# Train a categorical classifier to predict the Weather Type text labels
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train_cat)
joblib.dump(clf, 'weather_classifier.pkl')

unique_classes = sorted(list(set(y_categorical)))
class_to_idx = {name: i for i, name in enumerate(unique_classes)}

y_train_cat_num = np.array([class_to_idx[item] for item in y_train_cat.to_numpy()])
y_test_cat_num = np.array([class_to_idx[item] for item in y_test_cat.to_numpy()])

print("Model training results:")

# EVALUATING THE LINEAR REGRESSION Model
lr_train_preds = model.predict(X_train)
lr_test_preds = model.predict(X_test)

lr_test_idx_preds = np.clip(np.round(lr_test_preds * (len(unique_classes) - 1)), 0, len(unique_classes) - 1).astype(int)
lr_test_str_preds = np.array([unique_classes[idx] for idx in lr_test_idx_preds])

print("\n" + "="*50)
print("LINEAR REGRESSION PERFORMANCE Metrics")
print("="*50)
print(f" - Training Set Error (MAE)       : {mean_absolute_error(y_train, lr_train_preds):.4f}")
print(f" - Test Set Error (MAE)           : {mean_absolute_error(y_test, lr_test_preds):.4f}")
print(f" - Mean Squared Error (MSE)       : {mean_squared_error(y_test, lr_test_preds):.4f}")
print(f" - Root Mean Squared Error (RMSE) : {root_mean_squared_error(y_test, lr_test_preds):.4f}")
print(f" - R2 Performance Score           : {r2_score(y_test, lr_test_preds):.4f}")
print(f" - System Classification Accuracy : {accuracy_score(y_test_cat, lr_test_str_preds)*100:.2f}%")


# EVALUATING THE DECISION TREE CLASSIFIER Model
dt_train_str_preds = clf.predict(X_train)
dt_test_str_preds = clf.predict(X_test)

# Map categorical results back to numeric indices
dt_train_idx_preds = np.array([class_to_idx[item] for item in dt_train_str_preds])
dt_test_idx_preds = np.array([class_to_idx[item] for item in dt_test_str_preds])

dt_scaled_test_preds = dt_test_idx_preds / (len(unique_classes) - 1)
dt_scaled_train_preds = dt_train_idx_preds / (len(unique_classes) - 1)

print("\n" + "="*50)
print("DECISION TREE CLASSIFIER PERFORMANCE METRICS")
print("="*50)
print(f" - Training Set Error (MAE)       : {mean_absolute_error(y_train, dt_scaled_train_preds):.4f}")
print(f" - Test Set Error (MAE)           : {mean_absolute_error(y_test, dt_scaled_test_preds):.4f}")
print(f" - Mean Squared Error (MSE)       : {mean_squared_error(y_test, dt_scaled_test_preds):.4f}")
print(f" - Root Mean Squared Error (RMSE) : {root_mean_squared_error(y_test, dt_scaled_test_preds):.4f}")
print(f" - R2 Performance Score           : {r2_score(y_test, dt_scaled_test_preds):.4f}")
print(f" - System Classification Accuracy : {accuracy_score(y_test_cat, dt_test_str_preds)*100:.2f}%")

# Confusion Matrix Layout Render Matrix for Verification
print("\nDecision Tree Confusion Matrix Layout:")
cm = confusion_matrix(y_test_cat, dt_test_str_preds, labels=unique_classes)
cm_df = pd.DataFrame(cm, index=[f"Actual {x}" for x in unique_classes], columns=[f"Pred {x}" for x in unique_classes])
print(cm_df)

# Precision/recall metric arrays
print("\nAdvanced Metrics Matrix Breakdown (Precision and Recall):")
print(classification_report(y_test_cat, dt_test_str_preds, labels=unique_classes))

mse_train = mean_absolute_error(y_train, lr_train_preds)
mse_test = mean_absolute_error(y_test, lr_test_preds)
