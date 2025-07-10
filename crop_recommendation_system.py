import pandas as pd
from datetime import datetime

# Load Data 1 (crop suitability) and Data 3 (market demand and scarcity)
data1 = pd.read_csv("real data.csv")  # Data 1: Crop suitability
data3 = pd.read_csv("Maret_data.csv")  # Data 3: Market data


def check_crop_suitability(region, nitrogen, phosphorus, potassium, ph):
    region_crops = data1[data1["Region"] == region]
    suitable_crops = []

    for _, row in region_crops.iterrows():
        try:
            if (
                row["Min_N(kg/ha)"] <= nitrogen <= row["Max_N(kg/ha)"]
                and row["Min_P(mg/kg)"] <= phosphorus <= row["Max_P(mg/kg)"]
                and row["Min_K(mg/kg)"] <= potassium <= row["Max_K(mg/kg)"]
                and row["Min_pH"] <= ph <= row["Max_pH"]
            ):
                suitable_crops.append(
                    {
                        "Crop": row["Crop"],
                        "Growing_Period_Months": row.get(
                            "Growing_Period_Months", row.get("Min_Month", 3)
                        ),  # fallback
                    }
                )
        except KeyError as e:
            print(f"Missing column in data: {e}")
            continue
    return suitable_crops


def calculate_harvest_month(planting_month, growing_period):
    harvest_month = (planting_month + growing_period - 1) % 12
    return 12 if harvest_month == 0 else harvest_month


def check_market_viability(region, crop, harvest_month):
    market_data = data3[
        (data3["Region"] == region)
        & (data3["Crop"] == crop)
        & (data3["Month"] == harvest_month)
    ]

    if market_data.empty:
        print(
            f"Warning: No market data for {crop} in {region} for month {harvest_month}."
        )
        return False

    demand_level = market_data["Demand_Level"].iloc[0]
    scarcity = market_data["Scarcity_Period"].iloc[0]
    print(
        f"Market data for {crop} in {region}, month {harvest_month}: Demand={demand_level}, Scarcity={scarcity}"
    )
    return demand_level == "High" or scarcity == "Yes"


def find_alternative_crops(region, planting_month):
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
        return viable_crops
    else:
        return []


def recommend_crops(region, nitrogen, phosphorus, potassium, ph, planting_month):
    print(
        f"Checking crop suitability for {region} with N={nitrogen}, P={phosphorus}, K={potassium}, pH={ph}..."
    )
    suitable_crops = check_crop_suitability(region, nitrogen, phosphorus, potassium, ph)
    final_recommendations = []

    for crop_info in suitable_crops:
        crop = crop_info["Crop"]
        growing_period = int(crop_info["Growing_Period_Months"])
        harvest_month = calculate_harvest_month(planting_month, growing_period)
        if check_market_viability(region, crop, harvest_month):
            final_recommendations.append(
                {
                    "Crop": crop,
                    "Growing_Period_Months": growing_period,
                    "Harvest_Month": harvest_month,
                }
            )

    if not final_recommendations:
        print(
            "No directly suitable crops found with high market viability. Searching for alternatives..."
        )
        alternatives = find_alternative_crops(region, planting_month)
        if alternatives:
            print("Alternative crops with better market potential:")
            for alt in alternatives:
                print(alt)
        else:
            print("No alternative crops found.")
    else:
        print("Recommended crops based on soil and market data:")
        for rec in final_recommendations:
            print(rec)


if __name__ == "__main__":
    try:
        region = input("Enter region: ")
        nitrogen = float(input("Enter nitrogen level (kg/ha): "))
        phosphorus = float(input("Enter phosphorus level (mg/kg): "))
        potassium = float(input("Enter potassium level (mg/kg): "))
        ph = float(input("Enter soil pH: "))
        planting_month = int(input("Enter planting month (1-12): "))

        recommend_crops(region, nitrogen, phosphorus, potassium, ph, planting_month)
    except Exception as e:
        print(f"An error occurred: {e}")
