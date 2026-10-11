import os
import numpy as np
import rasterio
from scipy.ndimage import generic_filter


# ============================================================
# 1. PENGATURAN
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_TIF = os.path.join(
    BASE_DIR,
    "hasil_klasifikasi_jatim.tif"
)

OUTPUT_TIF = os.path.join(
    BASE_DIR,
    "hasil_klasifikasi_jatim_smooth.tif"
)


# ============================================================
# 2. BACA RASTER
# ============================================================

print("=" * 60)
print("SMOOTHING HASIL KLASIFIKASI")
print("=" * 60)

print("\nMembaca raster...")

with rasterio.open(INPUT_TIF) as src:

    data = src.read(1)

    profile = src.profile.copy()

    print(
        f"Ukuran raster: "
        f"{src.width} x {src.height}"
    )

    print(
        f"CRS: {src.crs}"
    )


# ============================================================
# 3. FUNGSI MAJORITY FILTER
# ============================================================

def majority_filter(values):

    values = values.astype(np.uint8)

    # Buang NoData
    values = values[values != 0]

    if len(values) == 0:
        return 0

    # Hitung jumlah setiap kelas
    counts = np.bincount(
        values,
        minlength=6
    )

    # Ambil kelas yang paling banyak
    return np.argmax(counts)


# ============================================================
# 4. SMOOTHING
# ============================================================

print("\nMelakukan majority filter 3x3...")
print("Mohon tunggu...")

smooth = generic_filter(
    data,
    majority_filter,
    size=3,
    mode="nearest"
)

smooth = smooth.astype(np.uint8)


# ============================================================
# 5. PERTAHANKAN NODATA
# ============================================================

smooth[data == 0] = 0


# ============================================================
# 6. SIMPAN HASIL
# ============================================================

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
        smooth,
        1
    )


# ============================================================
# 7. HASIL
# ============================================================

print("\n" + "=" * 60)
print("SMOOTHING SELESAI")
print("=" * 60)

print("\nFile hasil:")

print(
    OUTPUT_TIF
)

print("\nKelas yang tersedia:")

print("1 = Building")
print("2 = Farmland")
print("3 = Water")
print("4 = Forest")
print("5 = Mangrove")

print("=" * 60)