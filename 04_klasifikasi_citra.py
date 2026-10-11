import os
import glob
import joblib
import numpy as np
import rasterio


print("=" * 60)
print("KLASIFIKASI CITRA SENTINEL-2")
print("=" * 60)


# ============================================================
# 1. CARI FILE BAND
# ============================================================

print("\nMencari file TIFF...")

files = glob.glob("*.tiff") + glob.glob("*.tif")

band_files = {}

for file in files:
    nama = os.path.basename(file).lower()

    if "b04" in nama:
        band_files["B04"] = file

    elif "b08" in nama:
        band_files["B08"] = file

    elif "b8a" in nama:
        band_files["B8A"] = file

    elif "b11" in nama:
        band_files["B11"] = file


for band in ["B04", "B08", "B8A", "B11"]:
    if band not in band_files:
        raise FileNotFoundError(
            f"File {band} tidak ditemukan!"
        )

    print(f"{band}: {band_files[band]}")


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("\nMembaca model Random Forest...")

model = joblib.load(
    "random_forest_jatim.pkl"
)

print("Model berhasil dimuat.")


# ============================================================
# 3. BUKA BAND
# ============================================================

print("\nMembuka raster...")

src_b04 = rasterio.open(band_files["B04"])
src_b08 = rasterio.open(band_files["B08"])
src_b8a = rasterio.open(band_files["B8A"])
src_b11 = rasterio.open(band_files["B11"])

print(
    f"Ukuran raster: "
    f"{src_b04.width} x {src_b04.height}"
)

print(f"CRS: {src_b04.crs}")
print(f"Transform: {src_b04.transform}")


# ============================================================
# 4. BACA DATA BAND
# ============================================================

print("\nMembaca nilai pixel...")

b04 = src_b04.read(1).astype(np.float32)
b08 = src_b08.read(1).astype(np.float32)
b8a = src_b8a.read(1).astype(np.float32)
b11 = src_b11.read(1).astype(np.float32)


# ============================================================
# 5. CEK NO DATA
# ============================================================

nodata = src_b04.nodata

print(f"NoData B04: {nodata}")

if nodata is not None:

    valid = (
        (b04 != nodata) &
        (b08 != nodata) &
        (b8a != nodata) &
        (b11 != nodata)
    )

else:

    valid = np.ones(
        b04.shape,
        dtype=bool
    )


# ============================================================
# 6. SIAPKAN DATA PIXEL
# ============================================================

height, width = b04.shape

X = np.column_stack([
    b04.flatten(),
    b08.flatten(),
    b8a.flatten(),
    b11.flatten()
])

valid_flat = valid.flatten()

print(f"Total pixel: {len(X):,}")
print(f"Pixel valid: {valid_flat.sum():,}")


# ============================================================
# 7. PREDIKSI
# ============================================================

print("\nMelakukan klasifikasi...")

hasil = np.zeros(
    len(X),
    dtype=np.uint8
)

# Hanya pixel valid yang diprediksi
X_valid = X[valid_flat]

print(
    f"Pixel yang diklasifikasikan: "
    f"{len(X_valid):,}"
)

prediksi = model.predict(X_valid)


# ============================================================
# 8. KONVERSI CLASS KE ANGKA
# ============================================================

class_mapping = {
    "Building": 1,
    "Farmland": 2,
    "Water": 3,
    "Forest": 4,
    "Mangrove": 5
}

for nama_class, kode in class_mapping.items():

    mask = prediksi == nama_class

    hasil[
        np.where(valid_flat)[0][mask]
    ] = kode


# ============================================================
# 9. UBAH KE BENTUK RASTER
# ============================================================

hasil = hasil.reshape(
    height,
    width
)


# ============================================================
# 10. SIMPAN HASIL
# ============================================================

output = "hasil_klasifikasi_jatim.tif"

profile = src_b04.profile.copy()

profile.update(
    dtype=rasterio.uint8,
    count=1,
    compress="lzw",
    nodata=0
)

print("\nMenyimpan hasil...")

with rasterio.open(
    output,
    "w",
    **profile
) as dst:

    dst.write(
        hasil,
        1
    )


# ============================================================
# 11. STATISTIK HASIL
# ============================================================

print("\n" + "=" * 60)
print("HASIL KLASIFIKASI")
print("=" * 60)

for nama_class, kode in class_mapping.items():

    jumlah = np.sum(
        hasil == kode
    )

    print(
        f"{nama_class:10}: "
        f"{jumlah:,} pixel"
    )

print(
    f"\nNoData     : "
    f"{np.sum(hasil == 0):,} pixel"
)

print("\nFile hasil:")
print(output)

print("\n" + "=" * 60)
print("KLASIFIKASI SELESAI")
print("=" * 60)