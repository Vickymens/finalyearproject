import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load data
df = pd.read_csv("crop_growing_conditions.csv")
df.dropna(inplace=True)

# Backup year for later analysis
df["Original_Year"] = df["Year"]

# Set target and features
target_col = "Yield_t_ha"
exclude_cols = [target_col, "Notes", "Year"]
feature_cols = [col for col in df.columns if col not in exclude_cols]

# Encode categorical variables
categorical_cols = [col for col in feature_cols if df[col].dtype == "object"]
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Final features and target
X = df_encoded.drop([target_col, "Notes", "Year"], axis=1, errors="ignore")
y = df_encoded[target_col]
year_col = df_encoded["Original_Year"]  # For grouping later

# Train-test split
X_train, X_test, y_train, y_test, year_train, year_test = train_test_split(
    X, y, year_col, test_size=0.2, random_state=42
)

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models to tune
models_to_tune = {
    "Random Forest": (
        RandomForestRegressor(random_state=42),
        {"n_estimators": [100, 200], "max_depth": [None, 10]},
    ),
    "XGBoost": (
        XGBRegressor(random_state=42, verbosity=0),
        {"n_estimators": [100, 200], "max_depth": [3, 6], "learning_rate": [0.1, 0.3]},
    ),
    "Gradient Boosting": (
        GradientBoostingRegressor(random_state=42),
        {"n_estimators": [100, 200], "learning_rate": [0.1, 0.3], "max_depth": [3, 6]},
    ),
}

# Train and tune models
best_models = {}
results = []
predictions_df = pd.DataFrame()

for name, (model, param_grid) in models_to_tune.items():
    print(f"\nTuning {name}...")
    grid = GridSearchCV(model, param_grid, scoring="r2", cv=3, n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    best_model = grid.best_estimator_
    best_models[name] = best_model

    y_pred = best_model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results.append({"Model": name, "R² Score": r2, "MAE": mae, "RMSE": rmse})

    # Save predictions
    temp_df = pd.DataFrame(
        {"Actual": y_test, "Predicted": y_pred, "Model": name, "Year": year_test}
    )
    predictions_df = pd.concat([predictions_df, temp_df], ignore_index=True)

# Save predictions to CSV
predictions_df.to_csv("model_predictions.csv", index=False)
print("\nPredictions saved to 'model_predictions.csv'")

# Convert results to DataFrame
results_df = pd.DataFrame(results)
results_df["Accuracy (%)"] = results_df["R² Score"] * 100
print("\nModel Performance:")
print(results_df[["Model", "R² Score", "Accuracy (%)", "MAE", "RMSE"]])

# Save best model
best_idx = results_df["R² Score"].idxmax()
best_model_name = results_df.loc[best_idx, "Model"]
joblib.dump(best_models[best_model_name], "yield_prediction_model_new_one.joblib")
joblib.dump(scaler, "yield_prediction_scaler_new_one.joblib")
print(f"Best model '{best_model_name}' saved.")

# Plot results
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="Model", y="R² Score", palette="viridis")
plt.title("Model Comparison - R² Score")
plt.tight_layout()
plt.show()

# Boxplot of errors
plt.figure(figsize=(14, 7))
predictions_df["Absolute_Error"] = np.abs(
    predictions_df["Actual"] - predictions_df["Predicted"]
)
sns.boxplot(x="Model", y="Absolute_Error", data=predictions_df, palette="Set2")
plt.title("Absolute Error Distribution by Model")
plt.tight_layout()
plt.show()

# Analysis by Year
yearly_group = (
    predictions_df.groupby("Year")
    .agg({"Actual": "mean", "Predicted": "mean", "Absolute_Error": "mean"})
    .reset_index()
)
print("\nAverage Yield and Error by Year:")
print(yearly_group)

# Plot Yearly Trends
plt.figure(figsize=(12, 6))
plt.plot(yearly_group["Year"], yearly_group["Actual"], label="Actual", marker="o")
plt.plot(yearly_group["Year"], yearly_group["Predicted"], label="Predicted", marker="s")
plt.title("Average Yield by Year")
plt.xlabel("Year")
plt.ylabel("Yield (t/ha)")
plt.legend()
plt.tight_layout()
plt.show()
