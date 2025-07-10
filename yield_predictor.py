import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
df = pd.read_csv("crop_growing_conditions.csv")
print("Dataset Preview:")
print(df.head())

# Handle missing values if any
df.dropna(inplace=True)
print("Available columns:", df.columns.tolist())

# Define target and features
target_col = "Yield_t_ha"
exclude_cols = [target_col, "Notes"]  # Exclude target and notes from features
feature_cols = [col for col in df.columns if col not in exclude_cols]

# Automatically detect categorical columns (object or category dtype), except the target
categorical_cols = [col for col in feature_cols if df[col].dtype == "object"]
print("Categorical columns for one-hot encoding:", categorical_cols)

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Define features and target using all columns except the target and notes
X = df_encoded.drop([target_col, "Notes"], axis=1)
y = df_encoded[target_col]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "SVR": SVR(),
    "XGBoost": XGBRegressor(random_state=42),
}

# Store results
results = []

# Train and evaluate
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results.append({"Model": name, "R² Score": r2, "MAE": mae, "RMSE": rmse})

    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

# Convert to DataFrame for plotting
results_df = pd.DataFrame(results)
results_df["Accuracy (%)"] = results_df["R² Score"] * 100

# Print table of results including accuracy
print("\nModel Performance Summary:")
print(results_df[["Model", "R² Score", "Accuracy (%)", "MAE", "RMSE"]])

# Save the best model
best_idx = results_df["R² Score"].idxmax()
best_model_name = results_df.loc[best_idx, "Model"]
best_model = models[best_model_name]
print(
    f"\nBest model: {best_model_name} (R² Score: {results_df.loc[best_idx, 'R² Score']:.4f})"
)
joblib.dump(best_model, "yield_prediction_model.joblib")
joblib.dump(scaler, "yield_prediction_scaler.joblib")
print(
    "Best model and scaler saved as 'yield_prediction_model.joblib' and 'yield_prediction_scaler.joblib'."
)

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x="Model", y="R² Score", palette="viridis")
plt.title("Model Comparison - R² Score")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x="Model", y="MAE", palette="magma")
plt.title("Model Comparison - Mean Absolute Error (MAE)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x="Model", y="RMSE", palette="coolwarm")
plt.title("Model Comparison - Root Mean Squared Error (RMSE)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Additional graphical comparisons

# 1. Boxplot of absolute errors for each model
plt.figure(figsize=(14, 7))
abs_errors = []
model_names = []
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    abs_err = np.abs(y_test - y_pred)
    abs_errors.append(abs_err)
    model_names.extend([name] * len(abs_err))
abs_errors_flat = np.concatenate(abs_errors)
sns.boxplot(x=model_names, y=abs_errors_flat, palette="Set2")
plt.title("Absolute Error Distribution by Model")
plt.xlabel("Model")
plt.ylabel("Absolute Error")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 2. Scatter plot of predicted vs actual for each model
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for idx, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test_scaled)
    axes[idx].scatter(y_test, y_pred, alpha=0.6)
    axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    axes[idx].set_title(f"{name}")
    axes[idx].set_xlabel("Actual Yield")
    axes[idx].set_ylabel("Predicted Yield")
plt.suptitle("Predicted vs Actual Yield for Each Model")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
