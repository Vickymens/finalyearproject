import pandas as pd
import numpy as np

# Load the CSV
df = pd.read_csv("testing it again.csv")

# Set the rainfall range for Plantain only
min_rainfall_plantain = 1200
max_rainfall_plantain = 1500

# Find all Plantain rows
plantain_mask = df["Crop"].str.lower() == "plantain"

# Generate random rainfall values within the range for Plantain rows
np.random.seed(42)  # For reproducibility
random_rainfall_plantain = np.random.randint(
    min_rainfall_plantain, max_rainfall_plantain + 1, plantain_mask.sum()
)

# Assign the random rainfall values to Plantain rows
df.loc[plantain_mask, "Rainfall_mm"] = random_rainfall_plantain

# Save the updated CSV
df.to_csv("testing it again.csv", index=False)

print(
    "All Plantain rainfall values have been set to random values between 1200 and 1500."
)
