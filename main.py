import requests
import json


def get_weather_data(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return {
            "main_weather": data["weather"][0]["main"],
            "description": data["weather"][0]["description"],
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"]["speed"],
            "wind_direction": data["wind"]["deg"],
        }
    else:
        print("ERROR RESPONSE:")
        print(response.status_code)
        print(response.text)
        return f"Error: Unable to retrieve data for {city}."


# Sample usage
city = "Tarkwa,GH"  # Replace with any location of your choice
api_key = "2089648194dc69bb764b036ffd94f28b"  # Replace with your actual OpenWeatherMap API key

weather_data = get_weather_data(city, api_key)

if isinstance(weather_data, dict):
    print(f"Weather Data for {city}:")
    print(f"Main Weather: {weather_data['main_weather']}")
    print(f"Description: {weather_data['description']}")
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind Speed: {weather_data['wind_speed']} m/s")
    print(f"Wind Direction: {weather_data['wind_direction']}°")
else:
    print(weather_data)

    import requests

API_KEY = "2089648194dc69bb764b036ffd94f28b"
location = "Accra,GH"

geo_url = (
    f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={API_KEY}"
)
response = requests.get(geo_url)
data = response.json()

lat = data[0]["lat"]
lon = data[0]["lon"]

print("Latitude:", lat)
print("Longitude:", lon)

import pandas as pd

print("AI-Powered Crop Recommendation System Initialized")

# Load the crop nutrient and pH data
crop_data = pd.read_csv("crop_nutrients.csv")

# Simulate sensor data for soil nutrients and pH
# These values will be collected from sensors in the real system
soil_nitrogen = float(input("Enter soil Nitrogen (N) value: "))
soil_phosphorus = float(input("Enter soil Phosphorus (P) value: "))
soil_potassium = float(input("Enter soil Potassium (K) value: "))
soil_ph = float(input("Enter soil pH value: "))

# Display the sensor data to confirm
print("\nSensor data from soil:")
print(
    f"Nitrogen (N): {soil_nitrogen}, Phosphorus (P): {soil_phosphorus}, Potassium (K): {soil_potassium}, pH: {soil_ph}"
)


# Function to recommend suitable crops
def recommend_crops(soil_nitrogen, soil_phosphorus, soil_potassium, soil_ph):
    suitable_crops = []

    # Loop through each crop and compare its nutrient and pH ranges with soil data
    for index, row in crop_data.iterrows():
        crop_name = row["Crop"]
        min_nitrogen = row["Nitrogen_Min"]
        max_nitrogen = row["Nitrogen_Max"]
        min_phosphorus = row["Phosphorus_Min"]
        max_phosphorus = row["Phosphorus_Max"]
        min_potassium = row["Potassium_Min"]
        max_potassium = row["Potassium_Max"]
        min_ph = row["pH_Min"]
        max_ph = row["pH_Max"]

        # Check if the soil values are within the acceptable range for the crop
        if (
            min_nitrogen <= soil_nitrogen <= max_nitrogen
            and min_phosphorus <= soil_phosphorus <= max_phosphorus
            and min_potassium <= soil_potassium <= max_potassium
            and min_ph <= soil_ph <= max_ph
        ):
            suitable_crops.append(crop_name)

    return suitable_crops


# Get recommendations based on the soil data
recommended_crops = recommend_crops(
    soil_nitrogen, soil_phosphorus, soil_potassium, soil_ph
)

# Display the recommended crops
if recommended_crops:
    print("\nRecommended crops based on your soil data:")
    for crop in recommended_crops:
        print(f"- {crop}")
else:
    print("\nNo suitable crops found for the given soil conditions.")
    print("\nAdd more fertiliser to stabilise the soil.")
