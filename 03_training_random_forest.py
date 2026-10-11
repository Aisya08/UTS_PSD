import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


print("=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)


# ============================================================
# 1. BACA DATA TRAINING
# ============================================================

print("\nMembaca training_pixel.csv...")

df = pd.read_csv("training_pixel.csv")

print(f"Jumlah data : {len(df)}")
print(f"Kolom       : {list(df.columns)}")


# ============================================================
# 2. FITUR DAN TARGET
# ============================================================

features = [
    "B04",
    "B08",
    "B8A",
    "B11"
]

target = "class"

X = df[features]
y = df[target]


print("\nJumlah pixel setiap kelas:")
print(y.value_counts())


# ============================================================
# 3. BAGI DATA TRAINING DAN TESTING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nPembagian data:")
print(f"Data training : {len(X_train)}")
print(f"Data testing  : {len(X_test)}")


# ============================================================
# 4. RANDOM FOREST
# ============================================================

print("\nMembuat model Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


# ============================================================
# 5. TRAINING
# ============================================================

print("Mulai training...")

model.fit(X_train, y_train)

print("Training selesai.")


# ============================================================
# 6. PREDIKSI DATA TESTING
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 7. AKURASI
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("HASIL EVALUASI")
print("=" * 60)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Accuracy : {accuracy * 100:.2f}%")


# ============================================================
# 8. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

print("Confusion Matrix:")

labels = [
    "Building",
    "Farmland",
    "Water",
    "Forest",
    "Mangrove"
]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print(labels)
print(cm)


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

print("\nFeature Importance:")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)


# ============================================================
# 11. SIMPAN MODEL
# ============================================================

joblib.dump(
    model,
    "random_forest_jatim.pkl"
)

print("\nModel disimpan sebagai:")
print("random_forest_jatim.pkl")

print("\n" + "=" * 60)
print("TRAINING SELESAI")
print("=" * 60)