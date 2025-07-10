import joblib
import pandas as pd
from datetime import datetime
import calendar

# Load saved models and encoders
cls_model = joblib.load("testing_it_again_crop_selection_model.joblib")
reg_model = joblib.load("testing_it_again_yield_prediction_model.joblib")
le_crop = joblib.load("testing_it_again_crop_label_encoder.joblib")
scaler_cls = joblib.load("testing_it_again_cls_scaler.joblib")
scaler_reg = joblib.load("testing_it_again_reg_scaler.joblib")

# Load supporting CSVs
weather_df = pd.read_csv("Historical weather.csv")
growing_df = pd.read_csv("crop_growing_conditions.csv")
yield_df = pd.read_csv("testing it again.csv")  # For highest yield lookup
market_df = pd.read_csv("Market data.csv")  # New line for market data


# Helper for user input
def get_float(prompt, min_val, max_val):
    print(f"Range: {min_val} - {max_val}")
    while True:
        try:
            val = float(input(f"{prompt}: "))
            if val < min_val or val > max_val:
                print(f"Value should be between {min_val} and {max_val}.")
            else:
                return val
        except ValueError:
            print("Please enter a valid number.")


# Get user features except rainfall and temperature
user_features = {
    "N_percent": get_float("Enter Nitrogen percent", 0.04, 0.12),
    "P_mg_per_kg": get_float("Enter Phosphorus (mg/kg)", 5.0, 10.0),
    "K_cmol_per_kg": get_float("Enter Potassium (cmol/kg)", 0.3, 0.6),
    "pH": get_float("Enter soil pH", 5.0, 6.5),
}

# Get region and planting month
region = input("Enter your region (e.g., Volta): ").strip().capitalize()
month_input = input(
    "Enter planting month (1-12) or press Enter to use current month: "
).strip()
if month_input:
    planting_month = int(month_input)
else:
    planting_month = datetime.now().month

# Prepare results
crop_results = []

# Calculate rainfall tolerance ranges for each crop
rainfall_ranges = {}
for crop in yield_df["Crop"].unique():
    crop_rain = yield_df[yield_df["Crop"].str.lower() == crop.lower()]["Rainfall_mm"]
    min_rain = crop_rain.min()
    max_rain = crop_rain.max()
    min_tol = min_rain - 0.3 * min_rain
    max_tol = max_rain + 0.3 * max_rain
    rainfall_ranges[crop.lower()] = (min_tol, max_tol)

# For each crop, calculate rainfall and temperature for the growing period
for _, row in growing_df.iterrows():
    crop = row["Crop"]
    duration = int(row["Months"])  # Number of months to harvest

    # Calculate months for this crop
    months = [(planting_month + i - 1) % 12 + 1 for i in range(duration)]
    month_names = [
        calendar.month_name[m] for m in months
    ]  # ['January', 'February', ...]

    # Get weather data for region and months (case-insensitive)
    region_weather = weather_df[weather_df["Region"].str.lower() == region.lower()]
    period_weather = region_weather[region_weather["Month"].isin(month_names)]

    # Compute total rainfall and average temperature for the period
    total_rainfall = period_weather["Rainfall_mm"].sum()
    avg_temp = period_weather["Temp_C"].mean()

    # --- Rainfall tolerance check ---
    min_tol, max_tol = rainfall_ranges.get(crop.lower(), (None, None))
    if min_tol is not None and max_tol is not None:
        if not (min_tol <= total_rainfall <= max_tol):
            continue  # Skip this crop silently

    # Prepare input for prediction
    features = {
        "N_percent": user_features["N_percent"],
        "P_mg_per_kg": user_features["P_mg_per_kg"],
        "K_cmol_per_kg": user_features["K_cmol_per_kg"],
        "pH": user_features["pH"],
        "Temp_C": avg_temp,
        "Rainfall_mm": total_rainfall,
        "Crop_enc": le_crop.transform([crop])[0],
    }
    user_df = pd.DataFrame([features])

    # Predict yield
    features_reg = [
        "N_percent",
        "P_mg_per_kg",
        "K_cmol_per_kg",
        "pH",
        "Temp_C",
        "Rainfall_mm",
        "Crop_enc",
    ]
    user_reg_scaled = scaler_reg.transform(user_df[features_reg])
    pred_yield = reg_model.predict(user_reg_scaled)[0]

    # Get highest yield for this crop from yield_df
    crop_yields = yield_df[yield_df["Crop"].str.lower() == crop.lower()][
        "Yield_t_per_ha"
    ]
    if not crop_yields.empty:
        max_yield = crop_yields.max()
        pred_yield = min(pred_yield, max_yield)  # Cap first!
        yield_percent = (pred_yield / max_yield) * 100
    else:
        yield_percent = None

    crop_results.append(
        {
            "Crop": crop,
            "Predicted_Yield": round(pred_yield, 2),
            "Yield_Percent": round(yield_percent, 2)
            if yield_percent is not None
            else None,
            "Rainfall_mm": round(total_rainfall, 2)
            if total_rainfall is not None
            else None,
            "Temp_C": round(avg_temp, 2) if avg_temp is not None else None,
        }
    )

# Find all crops with yield percentage higher than 80%
if len(crop_results) == 1:
    best_crops = crop_results  # Only one crop, so it's the best by default
else:
    best_crops = [
        c
        for c in crop_results
        if c["Yield_Percent"] is not None and c["Yield_Percent"] > 80
    ]


def get_market_info(crop, region, harvest_month_name):
    # Try to match crop, region, and month (case-insensitive)
    row = market_df[
        (market_df["Crop"].str.lower() == crop.lower())
        & (market_df["Region"].str.lower() == region.lower())
        & (market_df["Month"].str.lower() == harvest_month_name.lower())
    ]
    if not row.empty:
        demand = row.iloc[0]["Demand"]
        recommendation = row.iloc[0]["Recommendation"]
        return demand, recommendation
    return None, None


print("\nBest crop(s) for your conditions (yield percentage > 80%):")
if best_crops:
    for c in best_crops:
        duration = int(
            growing_df[growing_df["Crop"].str.lower() == c["Crop"].lower()][
                "Months"
            ].values[0]
        )
        harvest_month_num = (
            planting_month + duration - 2
        ) % 12 + 1  # zero-based adjustment
        harvest_month_name = calendar.month_name[harvest_month_num]
        demand, recommendation = get_market_info(c["Crop"], region, harvest_month_name)
        print(
            f"{c['Crop']} - Predicted Yield: {c['Predicted_Yield']} tons/ha, Yield %: {c['Yield_Percent']}%"
        )
        print(f"  Rainfall used: {c['Rainfall_mm']} mm")
        print(f"  Temperature used: {c['Temp_C']} °C")
        print(f"  Harvest Month: {harvest_month_name}")
        if demand is not None:
            print(f"  Market Demand at harvest: {demand}")
            print(f"  Recommendation: {recommendation}")
        else:
            print("  No market data available for this crop at harvest.")
else:
    print("No crop has a yield percentage above 80% for your conditions.")

print("\nAll crop predictions:")
for c in crop_results:
    duration = int(
        growing_df[growing_df["Crop"].str.lower() == c["Crop"].lower()][
            "Months"
        ].values[0]
    )
    harvest_month_num = (planting_month + duration - 2) % 12 + 1
    harvest_month_name = calendar.month_name[harvest_month_num]
    demand, recommendation = get_market_info(c["Crop"], region, harvest_month_name)
    print(
        f"{c['Crop']}: {c['Predicted_Yield']} tons/ha, Yield %: {c['Yield_Percent']}%"
    )
    print(f"  Rainfall used: {c['Rainfall_mm']} mm")
    print(f"  Temperature used: {c['Temp_C']} °C")
    print(f"  Harvest Month: {harvest_month_name}")
    if demand is not None:
        print(f"  Market Demand at harvest: {demand}")
        print(f"  Recommendation: {recommendation}")
    else:
        print("  No market data available for this crop at harvest.")
