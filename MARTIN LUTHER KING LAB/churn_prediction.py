"""
Customer Churn Prediction with KNN
Author: Alejandro Torres De la Rocha
Description: Predict customer churn using KNN and the Telco Customer Churn dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================

DATA_DIR = Path("data")
CSV_FILE = DATA_DIR / "telco_churn.csv"

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {CSV_FILE}. "
        "Download the Telco Customer Churn CSV and save it as data/telco_churn.csv."
    )

df = pd.read_csv(CSV_FILE)

print("\n" + "=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)

print(f"Dataset shape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst few rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["Churn"].value_counts())


# ============================================================
# STEP 2: DATA PREPROCESSING
# ============================================================

print("\n" + "=" * 50)
print("DATA PREPROCESSING")
print("=" * 50)

df_clean = df.copy()

# Drop customer ID because it is an identifier, not a useful predictive feature.
if "customerID" in df_clean.columns:
    df_clean = df_clean.drop("customerID", axis=1)

# Convert TotalCharges to numeric if it exists.
# In this dataset, TotalCharges sometimes appears as text because of blank values.
if "TotalCharges" in df_clean.columns:
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")

# Fill numeric missing values with median.
numeric_columns = df_clean.select_dtypes(include=[np.number]).columns

for col in numeric_columns:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Fill categorical missing values with mode.
categorical_columns = df_clean.select_dtypes(include=["object"]).columns

for col in categorical_columns:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# Convert target variable Churn from Yes/No to 1/0.
df_clean["Churn"] = df_clean["Churn"].map({
    "Yes": 1,
    "No": 0
})

print("\nTarget after conversion:")
print(df_clean["Churn"].value_counts())

# One-hot encode categorical variables.
df_encoded = pd.get_dummies(df_clean, drop_first=True)

print("\nEncoded dataset shape:")
print(df_encoded.shape)

print("\nEncoded columns sample:")
print(df_encoded.columns[:20].tolist())


# ============================================================
# OPTIONAL VISUALIZATION
# ============================================================

print("\n" + "=" * 50)
print("VISUALIZATION")
print("=" * 50)

plt.figure(figsize=(6, 4))
df["Churn"].value_counts().plot(kind="bar")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("churn_distribution.png", dpi=150, bbox_inches="tight")
print("Saved visualization to churn_distribution.png")
plt.show()


# ============================================================
# STEP 3: SPLIT THE DATA
# ============================================================

X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 50)
print("DATA SPLIT")
print("=" * 50)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTest target distribution:")
print(y_test.value_counts())


# ============================================================
# IMPORTANT: SCALE FEATURES FOR KNN
# ============================================================

# KNN is distance-based, so scaling matters a lot.
# Without scaling, features with larger numbers can dominate the model.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# STEP 4: TRAIN KNN MODEL
# ============================================================

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

print("\nKNN model trained successfully with K=5.")

y_train_pred = knn.predict(X_train_scaled)
y_test_pred = knn.predict(X_test_scaled)


# ============================================================
# STEP 5: MAKE PREDICTIONS AND EVALUATE
# ============================================================

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_confusion = confusion_matrix(y_test, y_test_pred)

print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

print("\nConfusion Matrix:")
print(test_confusion)

print("\nFormatted Confusion Matrix:")
print("                  Predicted")
print("              Not Churn  Churn")
print(f"Actual Not Churn   {test_confusion[0, 0]:4d}   {test_confusion[0, 1]:4d}")
print(f"Actual Churn       {test_confusion[1, 0]:4d}   {test_confusion[1, 1]:4d}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=["Not Churn", "Churn"]))


# ============================================================
# STEP 6: EXPERIMENT WITH DIFFERENT K VALUES
# ============================================================

print("\n" + "=" * 50)
print("EXPERIMENTING WITH DIFFERENT K VALUES")
print("=" * 50)

k_values = [1, 3, 5, 7, 9, 11, 15]
results = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)

    y_pred_temp = knn_temp.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred_temp)
    prec = precision_score(y_test, y_pred_temp)
    rec = recall_score(y_test, y_pred_temp)

    results.append({
        "K": k,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec
    })

    print(f"K={k:2d}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")

results_df = pd.DataFrame(results)

best_k = results_df.loc[results_df["Accuracy"].idxmax(), "K"]
best_accuracy = results_df["Accuracy"].max()

print(f"\nBest K value based on accuracy: {best_k}")
print(f"Best accuracy: {best_accuracy:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(results_df["K"], results_df["Accuracy"], marker="o", label="Accuracy")
plt.plot(results_df["K"], results_df["Precision"], marker="s", label="Precision")
plt.plot(results_df["K"], results_df["Recall"], marker="^", label="Recall")
plt.xlabel("K Number of Neighbors")
plt.ylabel("Score")
plt.title("Customer Churn KNN Performance vs K Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("churn_knn_k_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved visualization to churn_knn_k_comparison.png")
plt.show()


# ============================================================
# STEP 7: ANALYSIS AND RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 50)
print("ANALYSIS AND RECOMMENDATIONS")
print("=" * 50)

print("""
The KNN model was able to predict customer churn using customer account, service, and demographic information.
For churn prediction, recall is especially important because the company wants to identify as many customers at risk of leaving as possible.
Precision is also important because contacting too many customers who are not actually at risk could waste company resources.

Based on this model, the company should focus retention efforts on customers who are predicted to churn.
Possible business actions include targeted discounts, service quality follow-ups, contract renewal offers, or personalized customer support.

Limitations:
- KNN can be sensitive to feature scaling, so StandardScaler was used.
- KNN does not directly explain which features are most important.
- The model may need more tuning or comparison with other models such as logistic regression, decision trees, or random forests.
- Accuracy alone can be misleading if the dataset is imbalanced, so precision and recall should also be reviewed.
""")
