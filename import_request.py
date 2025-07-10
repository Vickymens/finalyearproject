import requests
import json


def get_weather_data(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},GH&appid={api_key}&units=metric"
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


# Receive city from user
city_input = input("Enter city name (e.g., Tarkwa): ").strip()
city = city_input  # Only the city, 'GH' will be appended in the function

api_key = "2089648194dc69bb764b036ffd94f28b"  # Replace with your actual OpenWeatherMap API key

weather_data = get_weather_data(city, api_key)

if isinstance(weather_data, dict):
    print(f"Weather Data for {city},GH:")
    print(f"Main Weather: {weather_data['main_weather']}")
    print(f"Description: {weather_data['description']}")
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind Speed: {weather_data['wind_speed']} m/s")
    print(f"Wind Direction: {weather_data['wind_direction']}°")
else:
    print(weather_data)

# Get geo-coordinates for the same city entered by the user
API_KEY = api_key
location = f"{city},GH"

geo_url = (
    f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={API_KEY}"
)
response = requests.get(geo_url)
data = response.json()

lat = data[0]["lat"]
lon = data[0]["lon"]

print("Latitude:", lat)
print("Longitude:", lon)
