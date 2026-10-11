import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PENGATURAN
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT = os.path.join(
    BASE_DIR,
    "training_pixel.csv"
)


# ============================================================
# 2. BACA DATA
# ============================================================

print("=" * 60)
print("RANDOM FOREST - KLASIFIKASI TUTUPAN LAHAN")
print("=" * 60)

print("\nMembaca training_pixel.csv...")

df = pd.read_csv(INPUT)

print(f"Jumlah data: {len(df)}")
print("\nJumlah data per kelas:")
print(df["class"].value_counts())


# ============================================================
# 3. TENTUKAN FITUR DAN TARGET
# ============================================================

features = [
    "B04",
    "B08",
    "B8A",
    "B11"
]

X = df[features]
y = df["class"]


# ============================================================
# 4. BAGI DATA TRAINING DAN TESTING
# ============================================================

print("\nMembagi data 80% training dan 20% testing...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Data training : {len(X_train)}")
print(f"Data testing  : {len(X_test)}")


# ============================================================
# 5. BUAT MODEL RANDOM FOREST
# ============================================================

print("\nMembuat model Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# ============================================================
# 6. TRAINING
# ============================================================

print("Training model...")

model.fit(X_train, y_train)

print("Training selesai.")


# ============================================================
# 7. PREDIKSI
# ============================================================

print("\nMelakukan prediksi data testing...")

y_pred = model.predict(X_test)


# ============================================================
# 8. AKURASI
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("HASIL MODEL")
print("=" * 60)

print(
    f"\nAkurasi: {accuracy * 100:.2f}%"
)


# ============================================================
# 9. CLASSIFICATION REPORT
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
# 10. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

classes = model.classes_

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=classes
)

print(
    pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    )
)


# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

print("\nFeature Importance:")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print(importance.to_string(index=False))


# ============================================================
# 12. SELESAI
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST SELESAI")
print("=" * 60)