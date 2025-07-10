import pandas as pd
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load("yield_prediction_model.joblib")
scaler = joblib.load("yield_prediction_scaler.joblib")

# --- Example: Predict yield for new data ---

# 1. Prepare your new data as a DataFrame with the same features as used in training
# Example: Replace these values with your actual input
new_data = pd.DataFrame(
    [
        {
            # Fill in all features used for training, e.g.:
            "N_percent": 0.08,
            "P_mg_per_kg": 8.0,
            "K_cmol_per_kg": 0.45,
            "pH": 5.6,
            "Temp_C": 27.5,
            "Rainfall_mm": 580,
            # Add any other one-hot encoded columns (e.g. Crop, Region, Season, Soil_Type, etc.) if used in training
            # For one-hot columns, set to 1 or 0 as appropriate
            # Example: 'Region_Western': 1, 'Region_Eastern': 0, ...
        }
    ]
)

# 2. If you used get_dummies for categorical columns, make sure new_data columns match the training columns
# If needed, load your training columns and reindex:
# train_cols = list(pd.read_csv("new1.csv").columns)  # Or save feature_cols during training
# new_data = new_data.reindex(columns=train_cols, fill_value=0)

# 3. Scale the features
X_scaled = scaler.transform(new_data)

# 4. Predict (remember: model was trained on log1p(yield))
y_pred_log = model.predict(X_scaled)
y_pred = np.expm1(y_pred_log)  # Inverse of log1p

print(f"Predicted yield (t/ha): {y_pred[0]:.2f}")
