import pandas as pd
import numpy as np

# Define yield ranges for each crop
yield_ranges = {
    "maize": (1.24, 2.5),
    "rice": (1.8, 2.74),
    "cassava": (15.3, 22),
    "plantain": (9.38, 12.28),
    "cocoyam": (6.26, 8.79),
    "yam": (10.8, 13.94),
    "tomato": (28, 30),
    "pepper": (17, 20),
}

df = pd.read_csv("testing it again.csv")

# For each crop, assign yields based on sorted rainfall with 15-20% random deviation
for crop, (min_yield, max_yield) in yield_ranges.items():
    mask = df["Crop"].str.lower() == crop
    crop_df = df[mask].copy()
    if not crop_df.empty:
        # Sort by rainfall
        crop_df = crop_df.sort_values("Rainfall_mm").reset_index()
        n = len(crop_df)
        # Generate linearly spaced yields from min to max
        yields = np.linspace(min_yield, max_yield, n)
        # Add random deviation between -20% and +20%
        np.random.seed(42)  # For reproducibility
        deviation = np.random.uniform(-0.2, 0.2, n)
        yields_with_deviation = yields * (1 + deviation)
        # Clip yields to stay within min and max
        yields_with_deviation = np.clip(yields_with_deviation, min_yield, max_yield)
        # Round to 2 decimals
        yields_with_deviation = np.round(yields_with_deviation, 2)
        # Assign yields to the sorted DataFrame
        df.loc[crop_df["index"], "Yield_t_per_ha"] = yields_with_deviation

# Save the updated CSV
df.to_csv("testing it again.csv", index=False)
print(
    "Yield_t_per_ha values have been updated based on rainfall for each crop with 15-20% random deviation (rounded to 2 decimal places)."
)

# Get all unique crops
crops = df["Crop"].unique()

print("Maximum and minimum Yield_t_per_ha for each crop:\n")
for crop in crops:
    crop_yields = df[df["Crop"].str.lower() == crop.lower()]["Yield_t_per_ha"]
    max_yield = round(float(crop_yields.max()), 2)
    min_yield = round(float(crop_yields.min()), 2)
    print(f"{crop}: Max Yield = {max_yield}, Min Yield = {min_yield}")
