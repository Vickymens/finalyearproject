import requests
import json
import pandas as pd
from datetime import datetime


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

# Load Data 1
data1 = pd.read_csv("real data.csv")


def calculate_harvest_month(planting_month, growing_period):
    return ((int(planting_month) - 1 + int(growing_period) - 1) % 12) + 1


def suggest_adjustments(row, nitrogen, phosphorus, potassium, ph):
    adjustments = []
    if not (row["Min_N(kg/ha)"] <= nitrogen <= row["Max_N(kg/ha)"]):
        if nitrogen < row["Min_N(kg/ha)"]:
            adjustments.append(
                f"Add Nitrogen fertilizer to reach at least {row['Min_N(kg/ha)']} kg/ha."
            )
        else:
            adjustments.append(f"Reduce Nitrogen to below {row['Max_N(kg/ha)']} kg/ha.")
    if not (row["Min_P(mg/kg)"] <= phosphorus <= row["Max_P(mg/kg)"]):
        if phosphorus < row["Min_P(mg/kg)"]:
            adjustments.append(
                f"Add Phosphorus fertilizer to reach at least {row['Min_P(mg/kg)']} mg/kg."
            )
        else:
            adjustments.append(
                f"Reduce Phosphorus to below {row['Max_P(mg/kg)']} mg/kg."
            )
    if not (row["Min_K(mg/kg)"] <= potassium <= row["Max_K(mg/kg)"]):
        if potassium < row["Min_K(mg/kg)"]:
            adjustments.append(
                f"Add Potassium fertilizer to reach at least {row['Min_K(mg/kg)']} mg/kg."
            )
        else:
            adjustments.append(
                f"Reduce Potassium to below {row['Max_K(mg/kg)']} mg/kg."
            )
    if not (row["Min_pH"] <= ph <= row["Max_pH"]):
        if ph < row["Min_pH"]:
            adjustments.append(
                f"Raise soil pH to at least {row['Min_pH']} (apply lime)."
            )
        else:
            adjustments.append(
                f"Lower soil pH to below {row['Max_pH']} (apply sulfur)."
            )
    return adjustments


def recommend_crop(region, nitrogen, phosphorus, potassium, ph, planting_month=None):
    if planting_month in [None, 0]:
        planting_month = datetime.now().month

    region_data = data1[
        data1["Region"].str.strip().str.lower() == region.strip().lower()
    ]
    if region_data.empty:
        return f"No crop data available for region: {region}"

    # Score crops
    scored_crops = []
    for _, row in region_data.iterrows():
        crop = row["Crop"]
        min_n = row["Min_N(kg/ha)"]
        max_n = row["Max_N(kg/ha)"]
        min_p = row["Min_P(mg/kg)"]
        max_p = row["Max_P(mg/kg)"]
        min_k = row["Min_K(mg/kg)"]
        max_k = row["Max_K(mg/kg)"]
        min_ph = row["Min_pH"]
        max_ph = row["Max_pH"]
        min_month = row["Min_Month"]
        max_month = row["Max_Month"]
        notes = (
            row["Additional_note"]
            if "Additional_note" in row and pd.notnull(row["Additional_note"])
            else ""
        )

        n_suitable = min_n <= nitrogen <= max_n
        p_suitable = min_p <= phosphorus <= max_p
        k_suitable = min_k <= potassium <= max_k
        ph_suitable = min_ph <= ph <= max_ph

        score = sum([n_suitable, p_suitable, k_suitable, ph_suitable])

        scored_crops.append(
            {
                "Crop": crop,
                "Score": score,
                "Min_Month": min_month,
                "Max_Month": max_month,
                "Notes": notes,
                "row": row,
            }
        )

    # Recommend crops with score > 3
    suitable_crops = [c for c in scored_crops if c["Score"] > 3]
    output = []

    def month_name(month_number):
        """Return the month name for a given month number (1-12)."""
        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        if 1 <= month_number <= 12:
            return months[month_number - 1]
        return f"Month {month_number}"

    if suitable_crops:
        for crop in suitable_crops:
            output.append(f"Recommended Crop: {crop['Crop']}")
            output.append(f"Score: {crop['Score']}/4")
            output.append(f"Notes: {crop['Notes']}")
            harvest_months = []
            for growing_period in range(
                int(crop["Min_Month"]), int(crop["Max_Month"]) + 1
            ):
                harvest_month = calculate_harvest_month(planting_month, growing_period)
                harvest_months.append(month_name(harvest_month))
            # Show only the last 2 or 3 harvest months in a sentence
            if len(harvest_months) >= 2:
                last_months = (
                    harvest_months[-3:] if len(harvest_months) > 2 else harvest_months
                )
                if len(last_months) == 2:
                    harvest_str = f"{last_months[0]} and {last_months[1]}"
                else:
                    harvest_str = f"{', '.join(last_months[:-1])} to {last_months[-1]}"
                output.append(f"It will be ready for harvest around {harvest_str}.")
            else:
                output.append(f"It will be ready for harvest in {harvest_months[0]}.")
            if crop["Score"] < 4:
                adjustments = suggest_adjustments(
                    crop["row"], nitrogen, phosphorus, potassium, ph
                )
                if adjustments:
                    output.append("To improve suitability, do the following:")
                    for adj in adjustments:
                        output.append(f"- {adj}")
            output.append("")
        return "\n".join(output)

    # If none with score > 3, recommend crops with score > 2 and suggest adjustments
    fallback_crops = [c for c in scored_crops if c["Score"] > 2]
    if fallback_crops:
        for crop in fallback_crops:
            output.append(f"Alternative Crop: {crop['Crop']}")
            output.append(f"Score: {crop['Score']}/4")
            output.append(f"Notes: {crop['Notes']}")
            harvest_months = []
            for growing_period in range(
                int(crop["Min_Month"]), int(crop["Max_Month"]) + 1
            ):
                harvest_month = calculate_harvest_month(planting_month, growing_period)
                harvest_months.append(month_name(harvest_month))
            if len(harvest_months) >= 2:
                last_months = (
                    harvest_months[-3:] if len(harvest_months) > 2 else harvest_months
                )
                if len(last_months) == 2:
                    harvest_str = f"{last_months[0]} and {last_months[1]}"
                else:
                    harvest_str = f"{', '.join(last_months[:-1])} to {last_months[-1]}"
                output.append(f"It will be ready for harvest around {harvest_str}.")
            else:
                output.append(f"It will be ready for harvest in {harvest_months[0]}.")
            adjustments = suggest_adjustments(
                crop["row"], nitrogen, phosphorus, potassium, ph
            )
            if adjustments:
                output.append("To improve suitability, do the following:")
                for adj in adjustments:
                    output.append(f"- {adj}")
            output.append("")
        return "\n".join(output)

    return "No suitable crops found for your criteria and region."


# Example user input
region = input("Enter your region (e.g., Greater Accra, Central, etc.): ")
nitrogen = float(input("Enter soil Nitrogen level (kg/ha): "))
phosphorus = float(input("Enter soil Phosphorus level (mg/kg): "))
potassium = float(input("Enter soil Potassium level (mg/kg): "))
ph = float(input("Enter soil pH: "))
planting_month = input(
    "Enter planting month (1-12, or leave blank for current month): "
)
planting_month = int(planting_month) if planting_month.strip() else None

# Run recommendation
result = recommend_crop(region, nitrogen, phosphorus, potassium, ph, planting_month)
print(result)
