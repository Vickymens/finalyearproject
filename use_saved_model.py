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
import pandas as pd
from datetime import datetime

# Load Data 1 (crop suitability) and Data 3 (market demand and scarcity)
data1 = pd.read_csv("real data.csv")  # Data 1: Crop suitability
data3 = pd.read_csv("Maret_data.csv")  # Data 3: Market data


def check_crop_suitability(
    region, soil_n, soil_p, soil_k, soil_ph, rainfall, temperature
):
    """
    Check which crops are suitable based on soil, rainfall, and temperature (Data 1).
    Returns a list of suitable crops with their growing periods.
    """
    region_crops = data1[data1["Region"] == region]
    suitable_crops = []

    # Use the correct column names as in your CSV, e.g., 'Min_N(kg/ha)' instead of 'Min_N'
    for _, row in region_crops.iterrows():
        try:
            if (
                row["Min_N(kg/ha)"] <= soil_n <= row["Max_N(kg/ha)"]
                and row["Min_P(mg/kg)"] <= soil_p <= row["Max_P(mg/kg)"]
                and row["Min_K(mg/kg)"] <= soil_k <= row["Max_K(mg/kg)"]
                and row["Min_pH"] <= soil_ph <= row["Max_pH"]
                and row["Min_Rainfall"] <= rainfall <= row["Max_Rainfall"]
                and row["Min_Temperature"] <= temperature <= row["Max_Temperature"]
            ):
                suitable_crops.append(
                    {
                        "Crop": row["Crop"],
                        "Growing_Period_Months": row["Growing_Period_Months"],
                    }
                )
        except KeyError as e:
            print(f"Missing column in data: {e}")
            continue

    return suitable_crops


def calculate_harvest_month(planting_month, growing_period):
    """
    Calculate the harvest month based on planting month and growing period.
    Returns the harvest month (1–12).
    """
    harvest_month = (planting_month + growing_period - 1) % 12
    return 12 if harvest_month == 0 else harvest_month


def check_market_viability(region, crop, harvest_month):
    """
    Check market viability using Data 3 for a crop in a region at harvest month.
    Returns True if crop will sell (High demand or Scarcity), False otherwise.
    """
    market_data = data3[
        (data3["Region"] == region)
        & (data3["Crop"] == crop)
        & (data3["Month"] == harvest_month)
    ]

    if market_data.empty:
        print(
            f"Warning: No market data for {crop} in {region} for month {harvest_month}. Available crops in {region}: {data3[data3['Region'] == region]['Crop'].unique()}"
        )
        return False

    demand_level = market_data["Demand_Level"].iloc[0]
    scarcity = market_data["Scarcity_Period"].iloc[0]

    print(
        f"Market data for {crop} in {region}, month {harvest_month}: Demand={demand_level}, Scarcity={scarcity}"
    )
    return demand_level == "High" or scarcity == "Yes"


def find_alternative_crops(region, planting_month):
    """
    Find crops that would sell if planted today based on Data 3.
    Returns a list of crops with their growing periods and soil requirements.
    """
    # Dynamically detect the correct growing period column name
    growing_period_col = None
    for col in data1.columns:
        if col.replace("_", "").replace(" ", "").lower() in [
            "growingperiodmonths",
            "growingperiod",
            "growingperiod(months)",
            "growing_period_months",
            "min_month",
            "max_month",
        ]:
            growing_period_col = col
            break

    # If Min_Month and Max_Month exist, use them as the growing period range
    if "Min_Month" in data1.columns and "Max_Month" in data1.columns:
        region_crops = data1[data1["Region"] == region][
            ["Crop", "Min_Month", "Max_Month"]
        ].drop_duplicates()
        viable_crops = []
        for _, row in region_crops.iterrows():
            crop = row["Crop"]
            min_month = int(row["Min_Month"])
            max_month = int(row["Max_Month"])
            for growing_period in range(min_month, max_month + 1):
                harvest_month = calculate_harvest_month(planting_month, growing_period)
                if check_market_viability(region, crop, harvest_month):
                    crop_data = data1[
                        (data1["Region"] == region) & (data1["Crop"] == crop)
                    ].iloc[0]
                    viable_crops.append(
                        {
                            "Crop": crop,
                            "Growing_Period_Months": growing_period,
                            "Harvest_Month": harvest_month,
                            "Min_N": crop_data["Min_N(kg/ha)"],
                            "Max_N": crop_data["Max_N(kg/ha)"],
                            "Min_P": crop_data["Min_P(mg/kg)"],
                            "Max_P": crop_data["Max_P(mg/kg)"],
                            "Min_K": crop_data["Min_K(mg/kg)"],
                            "Max_K": crop_data["Max_K(mg/kg)"],
                            "Min_pH": crop_data["Min_pH"],
                            "Max_pH": crop_data["Max_pH"],
                        }
                    )
                    break  # Only add the first valid growing period for each crop
        return viable_crops

    # Otherwise, fall back to previous logic if Growing_Period_Months exists
    if growing_period_col and growing_period_col in data1.columns:
        region_crops = data1[data1["Region"] == region][
            ["Crop", growing_period_col]
        ].drop_duplicates()
        viable_crops = []
        for _, row in region_crops.iterrows():
            crop = row["Crop"]
            growing_period = row[growing_period_col]
            harvest_month = calculate_harvest_month(planting_month, growing_period)
            if check_market_viability(region, crop, harvest_month):
                crop_data = data1[
                    (data1["Region"] == region) & (data1["Crop"] == crop)
                ].iloc[0]
                viable_crops.append(
                    {
                        "Crop": crop,
                        "Growing_Period_Months": growing_period,
                        "Harvest_Month": harvest_month,
                        "Min_N": crop_data["Min_N(kg/ha)"],
                        "Max_N": crop_data["Max_N(kg/ha)"],
                        "Min_P": crop_data["Min_P(mg/kg)"],
                        "Max_P": crop_data["Max_P(mg/kg)"],
                        "Min_K": crop_data["Min_K(mg/kg)"],
                        "Max_K": crop_data["Max_K(mg/kg)"],
                        "Min_pH": crop_data["Min_pH"],
                        "Max_pH": crop_data["Max_pH"],
                    }
                )
        return viable_crops

    print(
        "Could not find a valid growing period column (e.g., 'Growing_Period_Months', 'Min_Month', 'Max_Month') in your data. Please check your CSV column names."
    )
    return []


def suggest_soil_adjustments(soil_n, soil_p, soil_k, soil_ph, crop_data):
    """
    Suggest soil adjustments for a crop based on current soil conditions and crop requirements.
    Returns a dictionary of recommendations.
    """
    adjustments = {}

    # Dynamically detect the correct keys for N, P, K in crop_data
    def get_key(d, options):
        for opt in options:
            if opt in d:
                return opt
        raise KeyError(f"None of {options} found in crop_data: {list(d.keys())}")

    n_key = get_key(crop_data, ["Min_N", "Min_N(kg/ha)"])
    n_max_key = get_key(crop_data, ["Max_N", "Max_N(kg/ha)"])
    p_key = get_key(crop_data, ["Min_P", "Min_P(mg/kg)"])
    p_max_key = get_key(crop_data, ["Max_P", "Max_P(mg/kg)"])
    k_key = get_key(crop_data, ["Min_K", "Min_K(mg/kg)"])
    k_max_key = get_key(crop_data, ["Max_K", "Max_K(mg/kg)"])

    # Use the detected keys for all checks
    if soil_n < crop_data[n_key]:
        adjustments[
            "Nitrogen"
        ] = f"Increase N by {crop_data[n_key] - soil_n:.2f} units (e.g., apply urea)."
    elif soil_n > crop_data[n_max_key]:
        adjustments[
            "Nitrogen"
        ] = f"Reduce N by {soil_n - crop_data[n_max_key]:.2f} units (e.g., reduce fertilizer)."
    if soil_p < crop_data[p_key]:
        adjustments[
            "Phosphorus"
        ] = f"Increase P by {crop_data[p_key] - soil_p:.2f} units (e.g., apply SSP)."
    elif soil_p > crop_data[p_max_key]:
        adjustments[
            "Phosphorus"
        ] = f"Reduce P by {soil_p - crop_data[p_max_key]:.2f} units (e.g., reduce fertilizer)."
    if soil_k < crop_data[k_key]:
        adjustments[
            "Potassium"
        ] = f"Increase K by {crop_data[k_key] - soil_k:.2f} units (e.g., apply MOP)."
    elif soil_k > crop_data[k_max_key]:
        adjustments[
            "Potassium"
        ] = f"Reduce K by {soil_k - crop_data[k_max_key]:.2f} units (e.g., reduce fertilizer)."
    if soil_ph < crop_data["Min_pH"]:
        adjustments[
            "pH"
        ] = f"Raise pH by {crop_data['Min_pH'] - soil_ph:.2f} units (e.g., apply lime)."
    elif soil_ph > crop_data["Max_pH"]:
        adjustments[
            "pH"
        ] = f"Lower pH by {soil_ph - crop_data['Max_pH']:.2f} units (e.g., apply sulfur)."
    return adjustments


def recommend_crops(
    region, soil_n, soil_p, soil_k, soil_ph, rainfall, temperature, planting_month
):
    """
    Recommend crops based on soil suitability (Data 1) and market viability (Data 3).
    If no suitable crops are viable, suggest alternatives with soil adjustments.
    """
    # Step 1: Find suitable crops
    suitable_crops = check_crop_suitability(
        region, soil_n, soil_p, soil_k, soil_ph, rainfall, temperature
    )

    if not suitable_crops:
        print(f"No crops are suitable for the given conditions in {region}.")
        print("Checking for alternative crops with market viability...")
    else:
        print(
            f"Suitable crops for {region}: {[crop['Crop'] for crop in suitable_crops]}"
        )

    # Step 2: Check market viability for suitable crops
    viable_crops = []
    for crop_info in suitable_crops:
        crop = crop_info["Crop"]
        growing_period = crop_info["Growing_Period_Months"]
        harvest_month = calculate_harvest_month(planting_month, growing_period)

        if check_market_viability(region, crop, harvest_month):
            viable_crops.append(
                {
                    "Crop": crop,
                    "Harvest_Month": harvest_month,
                    "Growing_Period_Months": growing_period,
                }
            )

    # Step 3: Recommend crops or suggest alternatives
    if viable_crops:
        print(f"Recommended crops for planting in {region} (market viable):")
        for crop in viable_crops[:2]:  # Limit to 1–2 crops
            print(
                f"- {crop['Crop']}: Harvest in month {crop['Harvest_Month']} "
                f"(after {crop['Growing_Period_Months']} months)"
            )
    else:
        print("No suitable crops are market viable at harvest time.")
        print("Finding alternative crops that would sell if planted today...")

        # Step 4: Find alternative crops
        alternative_crops = find_alternative_crops(region, planting_month)

        if alternative_crops:
            print("Alternative crops to plant today with market viability at harvest:")
            for crop in alternative_crops[:3]:  # Limit to 1–2 crops
                print(f"\nCrop: {crop['Crop']}")
                print(
                    f"Harvest Month: {crop['Harvest_Month']} (after {crop['Growing_Period_Months']} months)"
                )
                print("Soil Requirements:")
                print(f"- N: {crop['Min_N']}–{crop['Max_N']}")
                print(f"- P: {crop['Min_P']}–{crop['Max_P']}")
                print(f"- K: {crop['Min_K']}–{crop['Max_K']}")
                print(f"- pH: {crop['Min_pH']}–{crop['Max_pH']}")

                # Fix: pass the crop dict itself, which already has the correct keys
                adjustments = suggest_soil_adjustments(
                    soil_n, soil_p, soil_k, soil_ph, crop
                )
                if adjustments:
                    print("Soil Adjustments Needed:")
                    for param, advice in adjustments.items():
                        print(f"- {param}: {advice}")
                else:
                    print(
                        "No soil adjustments needed; current conditions are suitable."
                    )
        else:
            print(
                "No alternative crops with market viability found for this region and planting month."
            )


# Example usage (from your output)
region = "Volta"
soil_n = 50
soil_p = 20
soil_k = 150
soil_ph = 6.0
rainfall = 1200
temperature = 28
planting_month = 6  # June 2025

print(f"Running crop recommendation for {region}...")
recommend_crops(
    region, soil_n, soil_p, soil_k, soil_ph, rainfall, temperature, planting_month
)
