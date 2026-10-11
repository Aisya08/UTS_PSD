import os
import glob

import numpy as np
import pandas as pd
import rasterio

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. PENGATURAN
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAINING_CSV = os.path.join(
    BASE_DIR,
    "training_pixel.csv"
)

OUTPUT_TIF = os.path.join(
    BASE_DIR,
    "hasil_klasifikasi_jatim.tif"
)


# ============================================================
# 2. CARI FILE BAND OTOMATIS
# ============================================================

print("=" * 60)
print("KLASIFIKASI SELURUH AREA - RANDOM FOREST")
print("=" * 60)

print("\nMencari file TIFF...")

tif_files = glob.glob(
    os.path.join(BASE_DIR, "*.tif")
)

tif_files += glob.glob(
    os.path.join(BASE_DIR, "*.tiff")
)


BANDS = {}

for file in tif_files:

    nama = os.path.basename(file).upper()

    if "B04" in nama and "B04" not in BANDS:
        BANDS["B04"] = file

    elif "B08" in nama and "B08" not in BANDS:
        BANDS["B08"] = file

    elif "B8A" in nama and "B8A" not in BANDS:
        BANDS["B8A"] = file

    elif "B11" in nama and "B11" not in BANDS:
        BANDS["B11"] = file


print("\nBand yang ditemukan:")

for band in ["B04", "B08", "B8A", "B11"]:

    if band not in BANDS:
        raise FileNotFoundError(
            f"Band {band} tidak ditemukan."
        )

    print(
        f"{band}: {os.path.basename(BANDS[band])}"
    )


# ============================================================
# 3. BACA DATA TRAINING
# ============================================================

print("\nMembaca training_pixel.csv...")

df = pd.read_csv(TRAINING_CSV)

features = [
    "B04",
    "B08",
    "B8A",
    "B11"
]

X = df[features]
y = df["class"]


print(f"Jumlah pixel training: {len(df)}")

print("\nJumlah pixel setiap kelas:")
print(y.value_counts())


# ============================================================
# 4. TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X, y)

print("Training selesai.")


# ============================================================
# 5. BUKA RASTER
# ============================================================

print("\nMembuka raster...")

src_b04 = rasterio.open(BANDS["B04"])
src_b08 = rasterio.open(BANDS["B08"])
src_b8a = rasterio.open(BANDS["B8A"])
src_b11 = rasterio.open(BANDS["B11"])


print(
    f"Ukuran raster: "
    f"{src_b04.width} x {src_b04.height}"
)

print(
    f"CRS: {src_b04.crs}"
)

print(
    f"Resolusi: {src_b04.res}"
)


# ============================================================
# 6. BACA DATA BAND
# ============================================================

print("\nMembaca nilai pixel...")

b04 = src_b04.read(1)
b08 = src_b08.read(1)
b8a = src_b8a.read(1)
b11 = src_b11.read(1)


# ============================================================
# 7. BUAT MASK PIXEL VALID
# ============================================================

print("Mencari pixel valid...")

valid = (
    np.isfinite(b04) &
    np.isfinite(b08) &
    np.isfinite(b8a) &
    np.isfinite(b11)
)

print(
    f"Pixel valid: {valid.sum():,}"
)


# ============================================================
# 8. SIAPKAN DATA UNTUK PREDIKSI
# ============================================================

print("\nMenyiapkan data klasifikasi...")

rows, cols = b04.shape

pixels = np.column_stack([
    b04[valid],
    b08[valid],
    b8a[valid],
    b11[valid]
])


# ============================================================
# 9. PREDIKSI
# ============================================================

print("\nMelakukan klasifikasi...")
print("Mohon tunggu...")

prediction = model.predict(pixels)

print("Klasifikasi selesai.")


# ============================================================
# 10. KODE KELAS
# ============================================================
#
# 1 = Building
# 2 = Farmland
# 3 = Water
# 4 = Forest
# 5 = Mangrove
#

class_codes = {
    "Building": 1,
    "Farmland": 2,
    "Water": 3,
    "Forest": 4,
    "Mangrove": 5
}


print("\nKode kelas:")

for nama, kode in class_codes.items():
    print(f"{kode} = {nama}")


# ============================================================
# 11. UBAH NAMA KELAS MENJADI KODE
# ============================================================

predicted_codes = np.array([
    class_codes[kelas]
    for kelas in prediction
], dtype=np.uint8)


# ============================================================
# 12. BUAT RASTER HASIL
# ============================================================

print("\nMembuat raster hasil klasifikasi...")

hasil = np.zeros(
    (rows, cols),
    dtype=np.uint8
)

hasil[valid] = predicted_codes


# ============================================================
# 13. SIMPAN GEOTIFF
# ============================================================

profile = src_b04.profile.copy()

profile.update(
    dtype=rasterio.uint8,
    count=1,
    compress="lzw",
    nodata=0
)


with rasterio.open(
    OUTPUT_TIF,
    "w",
    **profile
) as dst:

    dst.write(
        hasil,
        1
    )


# ============================================================
# 14. TUTUP RASTER
# ============================================================

src_b04.close()
src_b08.close()
src_b8a.close()
src_b11.close()


# ============================================================
# 15. HASIL
# ============================================================

print("\n" + "=" * 60)
print("KLASIFIKASI SELESAI")
print("=" * 60)

print("\nFile hasil:")

print(
    OUTPUT_TIF
)

print("\nKode kelas:")

for nama, kode in class_codes.items():

    jumlah = np.sum(
        hasil == kode
    )

    print(
        f"{kode} - {nama}: "
        f"{jumlah:,} pixel"
    )

print("\nRaster 0 = NoData")

print("=" * 60)