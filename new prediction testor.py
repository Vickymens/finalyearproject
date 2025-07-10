import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("new1.csv")  # Replace with your actual filename

# Encode categorical features
le_crop = LabelEncoder()
le_region = LabelEncoder()
df["Crop"] = le_crop.fit_transform(df["Crop"])
df["Region"] = le_region.fit_transform(df["Region"])

# Define features and target
X = df.drop(columns=["Yield_t_per_ha"])
y = df["Yield_t_per_ha"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
}

# Train and evaluate each model
results = []
predictions_dict = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results.append((name, r2, rmse))
    predictions_dict[name] = preds

# Print results
print("Model Performance Comparison:")
print("{:<20} {:<10} {:<10}".format("Model", "R2 Score", "RMSE"))
for name, r2, rmse in sorted(results, key=lambda x: x[1], reverse=True):
    print(f"{name:<20} {r2:<10.4f} {rmse:<10.4f}")

# Prepare data for plotting
model_names = [x[0] for x in results]
r2_scores = [x[1] for x in results]
rmses = [x[2] for x in results]

# Plot R2 Scores
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(model_names, r2_scores, color="green")
plt.title("R² Score Comparison")
plt.ylabel("R² Score")
plt.xticks(rotation=45)

# Plot RMSE
plt.subplot(1, 2, 2)
plt.bar(model_names, rmses, color="red")
plt.title("RMSE Comparison")
plt.ylabel("Root Mean Squared Error")
plt.xticks(rotation=45)

# Create subplots
plt.figure(figsize=(18, 12))

# R² Score Bar Plot
plt.subplot(2, 2, 1)
plt.bar(model_names, r2_scores, color="green")
plt.title("R² Score Comparison")
plt.ylabel("R² Score")
plt.xticks(rotation=45)

# RMSE Bar Plot
plt.subplot(2, 2, 2)
plt.bar(model_names, rmses, color="red")
plt.title("RMSE Comparison")
plt.ylabel("Root Mean Squared Error")
plt.xticks(rotation=45)

# Predicted vs Actual Scatter Plot
plt.subplot(2, 2, 3)
for name in predictions_dict:
    plt.scatter(y_test, predictions_dict[name], label=name, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
plt.title("Predicted vs Actual Yield")
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.legend()

# Line Plot for Actual vs Predicted
plt.subplot(2, 2, 4)
y_test_sorted = y_test.reset_index(drop=True).sort_values().values
for name, preds in predictions_dict.items():
    sorted_preds = preds[np.argsort(y_test.reset_index(drop=True))]
    plt.plot(sorted_preds, label=name)
plt.plot(y_test_sorted, label="Actual", color="black", linewidth=2, linestyle="dashed")
plt.title("Actual vs Predicted Yields (Sorted)")
plt.xlabel("Sample Index (Sorted by Actual Yield)")
plt.ylabel("Yield")
plt.legend()

plt.tight_layout()
plt.show()
