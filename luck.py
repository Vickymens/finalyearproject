import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load the dataset
data = pd.read_csv("crop ata.csv")
print(data.columns)  # See actual column names

# Update these to match your file
features = ["N_percent", "P_mg_per_kg", "K_cmol_per_kg"]
target = "Crop"

# Drop missing values
data = data.dropna(subset=features + [target])

# Encode crop names to integers
le = LabelEncoder()
data["Crop_enc"] = le.fit_transform(data["Crop"])

# Train-test split
X = data[features]
y = data["Crop_enc"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Support Vector Machine": SVC(kernel="linear"),
}

accuracies = {}
reports = {}
conf_matrices = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracies[name] = accuracy
    reports[name] = classification_report(
        y_test, predictions, target_names=le.classes_, zero_division=0
    )
    conf_matrices[name] = confusion_matrix(y_test, predictions)

    print(f"--- {name} ---")
    print("Accuracy:", accuracy)
    print(reports[name])

# Plot accuracy comparison
plt.figure(figsize=(10, 5))
plt.bar(accuracies.keys(), accuracies.values(), color="skyblue")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.tight_layout()
plt.grid(axis="y")
plt.show()

# Confusion matrix heatmaps
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, (name, cm) in enumerate(conf_matrices.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", ax=axes[i])
    axes[i].set_title(f"Confusion Matrix - {name}")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")
# Hide any unused subplot axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# Distribution of nutrient values
plt.figure(figsize=(12, 5))
for idx, col in enumerate(features):
    plt.subplot(1, 3, idx + 1)
    sns.histplot(data[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# Pairplot to visualize relationship
# Use the original crop names for hue
data["Crop_name"] = le.inverse_transform(data["Crop_enc"])
sns.pairplot(data, hue="Crop_name", vars=features, palette="Set2")
plt.suptitle("Pairwise Plot of Nutrients by Crop", y=1.02)
plt.show()
