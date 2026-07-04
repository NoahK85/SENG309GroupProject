import pandas as pd
import csv

csvPd = pd.read_csv('weather_classification_data.csv')

# replace with whatever columns
columnsToNormalize = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)']

with open("weather_classification_data.csv", "r") as csvfile:
    # normalize
    for column in columnsToNormalize:
        # Vectorized calculation for min, max, and range
        normalMin = float(csvPd[column].min())
        normalMax = float(csvPd[column].max())
        normalRange = normalMax - normalMin

        print("Min:", normalMin)
        print("Max:", normalMax)
        print("Range:", normalRange)

        # Vectorized operations modify the whole column data block
        if normalRange != 0:
            csvPd[column] = (csvPd[column] - normalMin) / normalRange
        else:
            csvPd[column] = 0.0

        print(column, "normalized")

normalizedData = {
    'Temperature': csvPd['Temperature'].tolist(),
    'Humidity': csvPd['Humidity'].tolist(),
    'Wind Speed': csvPd['Wind Speed'].tolist(),
    'Precipitation (%)': csvPd['Precipitation (%)'].tolist(),
    'Cloud Cover': csvPd['Cloud Cover'].tolist(),
    'Atmospheric Pressure': csvPd['Atmospheric Pressure'].tolist(),
    'UV Index': csvPd['UV Index'].tolist(),
    'Season': csvPd['Season'].tolist(),
    'Visibility (km)': csvPd['Visibility (km)'].tolist() if 'Visibility (km)' in csvPd else csvPd.iloc[:, 8].tolist(),
    'Location': csvPd['Location'].tolist(),
    'Weather Type': csvPd['Weather Type'].tolist()
}

# export new file
with open("normalized_weather_data.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(normalizedData.keys())

    for column in normalizedData:
        normalizedData[column] = iter(normalizedData[column])

    while True:
        row = [next(normalizedData[column], "") for column in normalizedData]
        if all(val == "" for val in row):
            break
        w.writerow(row)

print("CSV normalized")
