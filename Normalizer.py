import pandas as pd
import csv

csvPd = pd.read_csv('weather_classification_data.csv')

#replace with whatever columns
columnsToNormalize = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)']

#prepare normalizeddata
normalizedData = {
            'Temperature'          : [],
            'Humidity'             : [],
            'Wind Speed'           : [],
            'Precipitation (%)'    : [],
            'Cloud Cover'          : [],
            'Atmospheric Pressure' : [],
            'UV Index'             : [],
            'Season'               : [],
            'Visibility (km)'      : [],
            'Location'             : [],
            'Weather Type'         : []
       
        }
with open("weather_classification_data.csv", "r") as csvfile:
    for row in range(csvPd['Temperature'].size):
            normalizedData.get('Temperature').append(csvPd.get('Temperature')[row])
            normalizedData.get('Humidity').append(csvPd.get('Humidity')[row])
            normalizedData.get('Wind Speed').append(csvPd.get('Wind Speed')[row])
            normalizedData.get('Precipitation (%)').append(csvPd.get('Precipitation (%)')[row])
            normalizedData.get('Cloud Cover').append(csvPd.get('Cloud Cover')[row])
            normalizedData.get('Atmospheric Pressure').append(csvPd.get('Atmospheric Pressure')[row])
            normalizedData.get('UV Index').append(csvPd.get('UV Index')[row])
            normalizedData.get('Season').append(csvPd.get('Season')[row])
            normalizedData.get('Visibility (km)').append(csvPd.get('Visibility (km)')[row])
            normalizedData.get('Location').append(csvPd.get('Location')[row])
            normalizedData.get('Weather Type').append(csvPd.get('Weather Type')[row])
    
    #normalize
    for column in columnsToNormalize:
        #get min and max
        normalMin = float(csvPd[column][0])
        normalMax = float(csvPd[column][0])
        for row in csvPd[column]:
            rowData = float(row)
            if rowData > normalMax:
                normalMax = rowData
            if rowData < normalMin:
                normalMin = rowData
        normalRange = normalMax - normalMin
        print("Min:", normalMin)
        print("Max:", normalMax)
        print("Range:", normalRange)
        
        i = 0
        #actually normalize
        for row in csvPd[column]:
            rowData = float(row)
            rowData = (rowData - normalMin) / normalRange
            normalizedData.get(column)[i] = rowData
            i += 1
        print (column, "normalized")
        
#export new file
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
