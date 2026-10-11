import os

import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ============================================================
# 1. PENGATURAN
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_TIF = os.path.join(
    BASE_DIR,
    "hasil_klasifikasi_jatim.tif"
)

OUTPUT_PNG = os.path.join(
    BASE_DIR,
    "peta_hasil_klasifikasi.png"
)


# ============================================================
# 2. BACA RASTER
# ============================================================

print("=" * 60)
print("VISUALISASI HASIL KLASIFIKASI")
print("=" * 60)

print("\nMembaca raster hasil...")

with rasterio.open(INPUT_TIF) as src:

    data = src.read(1)

    bounds = src.bounds

    print(f"Ukuran raster: {src.width} x {src.height}")
    print(f"CRS: {src.crs}")


# ============================================================
# 3. WARNA KELAS
# ============================================================

# 0 = NoData
# 1 = Building
# 2 = Farmland
# 3 = Water
# 4 = Forest
# 5 = Mangrove

colors = [
    "white",
    "red",
    "yellow",
    "blue",
    "green",
    "darkgreen"
]

cmap = ListedColormap(colors)


# ============================================================
# 4. TAMPILKAN PETA
# ============================================================

print("\nMembuat visualisasi...")

fig, ax = plt.subplots(
    figsize=(12, 10)
)

image = ax.imshow(
    data,
    cmap=cmap,
    vmin=0,
    vmax=5,
    extent=[
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top
    ]
)


# ============================================================
# 5. JUDUL DAN LABEL
# ============================================================

ax.set_title(
    "Peta Klasifikasi Tutupan Lahan Jawa Timur",
    fontsize=16
)

ax.set_xlabel(
    "Longitude"
)

ax.set_ylabel(
    "Latitude"
)


# ============================================================
# 6. LEGEND
# ============================================================

from matplotlib.patches import Patch

legend_elements = [

    Patch(
        facecolor="red",
        label="Building"
    ),

    Patch(
        facecolor="yellow",
        label="Farmland"
    ),

    Patch(
        facecolor="blue",
        label="Water"
    ),

    Patch(
        facecolor="green",
        label="Forest"
    ),

    Patch(
        facecolor="darkgreen",
        label="Mangrove"
    )
]

ax.legend(
    handles=legend_elements,
    loc="upper right",
    title="Kelas Tutupan Lahan"
)


# ============================================================
# 7. GRID
# ============================================================

ax.grid(
    True,
    alpha=0.3
)


# ============================================================
# 8. SIMPAN GAMBAR
# ============================================================

plt.tight_layout()

plt.savefig(
    OUTPUT_PNG,
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 9. SELESAI
# ============================================================

print("\n" + "=" * 60)
print("VISUALISASI SELESAI")
print("=" * 60)

print("\nFile gambar:")
print(OUTPUT_PNG)