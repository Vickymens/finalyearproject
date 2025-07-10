import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    classification_report,
)
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("new1.csv")

# Print crop distribution to check for imbalance
print("Crop distribution:\n", df["Crop"].value_counts())

# Encode categorical columns
le_crop = LabelEncoder()
le_region = LabelEncoder()
df["Crop_enc"] = le_crop.fit_transform(df["Crop"])
df["Region_enc"] = le_region.fit_transform(df["Region"])

# Features for crop selection (classification)
features_cls = [
    "Region_enc",
    "N_percent",
    "P_mg_per_kg",
    "K_cmol_per_kg",
    "pH",
    "Temp_C",
    "Rainfall_mm",
]
target_cls = "Crop_enc"

X_cls = df[features_cls]
y_cls = df[target_cls]

# Scale features for classification (helps SVM/KNN, doesn't hurt trees)
scaler_cls = StandardScaler()
X_cls_scaled = scaler_cls.fit_transform(X_cls)

# Stratified split for balanced classes
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
    X_cls_scaled, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

# Try both RandomForest and DecisionTree for crop selection
crop_models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}
for name, model in crop_models.items():
    model.fit(X_cls_train, y_cls_train)
    y_pred = model.predict(X_cls_test)
    acc = accuracy_score(y_cls_test, y_pred)
    print(f"\n{name} Crop Selection Accuracy: {acc:.3f}")
    print(classification_report(y_cls_test, y_pred, target_names=le_crop.classes_))

# Choose the best crop model (Random Forest by default)
crop_model = crop_models["Random Forest"]

# Features for yield prediction (regression)
# Use the predicted crop, not the original label
features_reg = [
    "Region_enc",
    "Crop_enc",
    "N_percent",
    "P_mg_per_kg",
    "K_cmol_per_kg",
    "pH",
    "Temp_C",
    "Rainfall_mm",
]
target_reg = "Yield_t_per_ha"

X_reg = df[features_reg]
y_reg = df[target_reg]

# Scale features for regression
scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)

# Split regression data
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg_scaled, y_reg, test_size=0.2, random_state=42
)

# Use RandomForest for regression (best for tabular data)
reg_model = RandomForestRegressor(random_state=42)
reg_model.fit(X_reg_train, y_reg_train)
y_reg_pred = reg_model.predict(X_reg_test)
r2 = r2_score(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
print(f"\nYield Prediction R2: {r2:.3f}, RMSE: {rmse:.3f}")

# ----------------------------
# TEST PREDICTION
# ----------------------------
# Example: Predict crop + yield for new input
sample_input = pd.DataFrame(
    [
        {
            "Region_enc": le_region.transform(["Western"])[0],
            "N_percent": 0.08,
            "P_mg_per_kg": 8.0,
            "K_cmol_per_kg": 0.45,
            "pH": 6.5,
            "Temp_C": 27.5,
            "Rainfall_mm": 700,
        }
    ]
)

# Scale sample input for crop selection
sample_input_scaled = scaler_cls.transform(sample_input)

# Predict crop
predicted_crop_int = crop_model.predict(sample_input_scaled)[0]
predicted_crop_name = le_crop.inverse_transform([predicted_crop_int])[0]
print("\nPredicted Suitable Crop:", predicted_crop_name)

# Add predicted crop to input for yield prediction
sample_input["Crop_enc"] = predicted_crop_int

# Scale for regression (must match training order)
sample_input_reg = sample_input[
    [
        "Region_enc",
        "Crop_enc",
        "N_percent",
        "P_mg_per_kg",
        "K_cmol_per_kg",
        "pH",
        "Temp_C",
        "Rainfall_mm",
    ]
]
sample_input_reg_scaled = scaler_reg.transform(sample_input_reg)

# Predict yield
predicted_yield = reg_model.predict(sample_input_reg_scaled)[0]
print("Predicted Yield (tons/ha):", round(predicted_yield, 2))
