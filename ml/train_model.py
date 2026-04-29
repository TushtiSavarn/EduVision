# =========================
# 📦 IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import joblib

# =========================
# 📂 LOAD DATASET
# =========================

# ⚠️ IMPORTANT: Ensure dataset.csv is in SAME folder
df = pd.read_csv("dataset.csv")

print("Dataset Loaded Successfully!\n")
print(df.head())

# =========================
# 🧹 DATA CLEANING
# =========================

required_columns = ["attendance", "periodical", "assignment", "label"]
df = df[required_columns]

df.dropna(inplace=True)

# Clip values
df["attendance"] = df["attendance"].clip(0, 100)
df["periodical"] = df["periodical"].clip(0, 100)
df["assignment"] = df["assignment"].clip(0, 100)

# =========================
# 📊 LABEL DISTRIBUTION
# =========================

plt.figure()
sns.countplot(x="label", data=df)
plt.title("Label Distribution")
plt.show()

# =========================
# 🔥 FEATURES
# =========================

X = df[["attendance", "periodical", "assignment"]]
y = df["label"]

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\nLabel Mapping:")
for i, label in enumerate(encoder.classes_):
    print(f"{label} -> {i}")

# =========================
# 📉 CORRELATION
# =========================

plt.figure()
sns.heatmap(X.corr(), annot=True)
plt.title("Feature Correlation")
plt.show()

# =========================
# ✂️ TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# =========================
# 🌲 MODEL
# =========================

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train, y_train)

# =========================
# 📊 EVALUATION
# =========================

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# =========================
# 🔲 CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =========================
# 🔁 CROSS VALIDATION
# =========================

cv_scores = cross_val_score(rf_model, X, y_encoded, cv=5)

print("\nCross Validation Accuracy:",
      round(cv_scores.mean() * 100, 2), "%")

# =========================
# 📌 FEATURE IMPORTANCE
# =========================

feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n")
print(feature_df)

plt.figure()
sns.barplot(x="Importance", y="Feature", data=feature_df)
plt.title("Feature Importance")
plt.show()

# =========================
# ⚔️ MODEL COMPARISON
# =========================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(),
    "Random Forest": rf_model
}

print("\nModel Comparison:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"{name}: {round(acc * 100, 2)} %")

# =========================
# 🧪 SAMPLE TEST
# =========================

print("\nSample Prediction Test")

sample = X_test.iloc[0].values.reshape(1, -1)

predicted = rf_model.predict(sample)

print("Features:", X_test.iloc[0].to_dict())
print("Predicted:", encoder.inverse_transform(predicted)[0])

# =========================
# 💾 SAVE MODEL
# =========================

joblib.dump(rf_model, "student_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")

print("\nModel Saved Successfully!")

