import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    r2_score,
    mean_squared_error,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("Final data selection 1.csv")

# Features for crop selection and yield prediction
features = ["N_percent", "P_mg_per_kg", "K_cmol_per_kg", "pH", "Temp_C", "Rainfall_mm"]

# Encode crop labels
le_crop = LabelEncoder()
df["Crop_enc"] = le_crop.fit_transform(df["Crop"])
X_cls = df[features]
y_cls = df["Crop_enc"]

# Check for class imbalance
print("Crop distribution:\n", df["Crop"].value_counts())

# Scale features
scaler_cls = StandardScaler()
X_cls_scaled = scaler_cls.fit_transform(X_cls)

# Stratified split for classification
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_cls_scaled, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

# Define 5 models for crop selection
models_cls = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
}

cls_results = []
fitted_classifiers = {}
for name, model in models_cls.items():
    model.fit(Xc_train, yc_train)
    y_pred = model.predict(Xc_test)
    acc = accuracy_score(yc_test, y_pred)
    cls_results.append({"Model": name, "Accuracy": acc})
    fitted_classifiers[name] = model
    print(f"\n{name} Crop Selection Accuracy: {acc:.3f}")
    print(classification_report(yc_test, y_pred, target_names=le_crop.classes_))

# Plot classification accuracy
cls_df = pd.DataFrame(cls_results)
plt.figure(figsize=(8, 5))
sns.barplot(data=cls_df, x="Model", y="Accuracy", palette="viridis")
plt.title("Crop Selection Model Accuracy Comparison")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Show confusion matrix for the best model
best_cls_row = max(cls_results, key=lambda x: x["Accuracy"])
best_cls_model = fitted_classifiers[best_cls_row["Model"]]
from sklearn.metrics import confusion_matrix

y_pred_best = best_cls_model.predict(Xc_test)
cm = confusion_matrix(yc_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le_crop.classes_,
    yticklabels=le_crop.classes_,
)
plt.xlabel("Predicted Crop")
plt.ylabel("Actual Crop")
plt.title(f"Confusion Matrix - {best_cls_row['Model']}")
plt.tight_layout()
plt.show()

# ----------------- Yield Prediction Section -----------------
# Add Crop_enc as a feature for regression
features_reg = features + ["Crop_enc"]
X_reg = df[features_reg]
y_reg = df["Yield_t_per_ha"]

scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg_scaled, y_reg, test_size=0.2, random_state=42
)

regressors = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Linear Regression": LinearRegression(),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
}

reg_results = []
fitted_regressors = {}
for name, model in regressors.items():
    model.fit(Xr_train, yr_train)
    y_pred = model.predict(Xr_test)
    r2 = r2_score(yr_test, y_pred)
    rmse = np.sqrt(mean_squared_error(yr_test, y_pred))
    reg_results.append({"Model": name, "R2": r2, "RMSE": rmse})
    fitted_regressors[name] = model
    print(f"\n{name} Yield Prediction R2: {r2:.3f}, RMSE: {rmse:.2f}")

# Plot regression R2
reg_df = pd.DataFrame(reg_results)
plt.figure(figsize=(8, 5))
sns.barplot(data=reg_df, x="Model", y="R2", palette="mako")
plt.title("Yield Prediction Model R2 Score Comparison")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Show RMSE for regression models
plt.figure(figsize=(8, 5))
sns.barplot(data=reg_df, x="Model", y="RMSE", palette="rocket")
plt.title
plt.tight_layout()
plt.show()
