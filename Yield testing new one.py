import pandas as pd
import joblib

# Load the saved model and scaler
model = joblib.load("yield_prediction_model_new_one.joblib")
scaler = joblib.load("yield_prediction_scaler_new_one.joblib")

# Load the columns used for training (from the original data)
original_df = pd.read_csv("crop_growing_conditions.csv")
target_col = "Yield_t_ha"
exclude_cols = [target_col, "Notes"]
feature_cols = [col for col in original_df.columns if col not in exclude_cols]
categorical_cols = [col for col in feature_cols if original_df[col].dtype == "object"]

# Prepare new data for prediction (replace with your actual input)
# Example:
# new_data = pd.DataFrame([{
#     "Region": "Greater Accra",
#     "Crop": "Maize",
#     "Planting_Month": 4,
#     "Season": "Major Rainy (Apr–Jul)",
#     "Year": 2025
# }])

# Show available options for categorical columns
for col in categorical_cols:
    unique_vals = original_df[col].unique()
    print(f"Available options for {col}: {list(unique_vals)}")

# Month mapping info for user
print("For Planting_Month, use numbers: January=1, February=2, ..., December=12")

# Prompt user for input with examples
new_data = {}
for col in feature_cols:
    if col in categorical_cols:
        example = f" (e.g., {original_df[col].unique()[0]})"
        new_data[col] = input(f"Enter {col}{example}: ")
    else:
        if col.lower() == "planting_month":
            new_data[col] = float(input(f"Enter {col} (1=Jan, 2=Feb, ..., 12=Dec): "))
        else:
            new_data[col] = float(input(f"Enter {col}: "))
new_data = pd.DataFrame([new_data])

# One-hot encode new data to match training columns
df_encoded = pd.get_dummies(original_df, columns=categorical_cols)
all_columns = df_encoded.drop([target_col, "Notes"], axis=1).columns
new_data_encoded = pd.get_dummies(new_data, columns=categorical_cols)
new_data_encoded = new_data_encoded.reindex(columns=all_columns, fill_value=0)

# Scale features
new_data_scaled = scaler.transform(new_data_encoded)

# Predict
predicted_yield = model.predict(new_data_scaled)
print("Predicted Yield (t/ha):", predicted_yield[0])
