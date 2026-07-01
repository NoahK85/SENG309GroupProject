#from sklearn.externals
import joblib

#get values from input
print("Welcome to the precipitation predictor")
print(" Please input temperature (degrees celsius): ", end="")
inputTemp = float(input())
print(" Please input humidity (%): ", end="")
inputHum = float(input())
print(" please input wind speed (mph): ", end="")
inputWind = float(input())
print("Predicting precipitation...")

# Load our trained model
model = joblib.load('weather_data_model.pkl')
# Define the weather that we want to value (with the values in the same order as in the training data)
weather = [
(inputTemp + 25) / 134,  # temp
(inputHum  - 20) / 89,   # humidity
(inputWind)      / 48.5, # wind speed
]
 
weathers = [
weather
]
# Make a prediction for each weather in the weathers array (we only have one)
precipitation_values = model.predict(weathers)

predicted_value = precipitation_values[0]
# Print the results
print("House details:")
print(f"- {-25 + weather[0] * 134:,.5f} temperature")
print(f"- {20 +  weather[1] * 89:,.5f} humidity")
print(f"- {weather[2] * 48.5:,.5} wind speed")
print(f"Estimated precipitation: {predicted_value * 100:,.2f}%")