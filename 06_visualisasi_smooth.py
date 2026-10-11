import os

import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_TIF = os.path.join(
    BASE_DIR,
    "hasil_klasifikasi_jatim_smooth.tif"
)

OUTPUT_PNG = os.path.join(
    BASE_DIR,
    "peta_hasil_klasifikasi_smooth.png"
)


print("=" * 60)
print("VISUALISASI HASIL SMOOTHING")
print("=" * 60)

with rasterio.open(INPUT_TIF) as src:

    data = src.read(1)
    bounds = src.bounds

    print(f"Ukuran raster: {src.width} x {src.height}")
    print(f"CRS: {src.crs}")


# 0 = NoData
# 1 = Building
# 2 = Farmland
# 3 = Water
# 4 = Forest
# 5 = Mangrove

cmap = ListedColormap([
    "white",
    "red",
    "yellow",
    "blue",
    "green",
    "darkgreen"
])


fig, ax = plt.subplots(
    figsize=(12, 10)
)

ax.imshow(
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


ax.set_title(
    "Peta Klasifikasi Tutupan Lahan Jawa Timur - Smoothed",
    fontsize=16
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")


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

ax.grid(
    True,
    alpha=0.3
)


plt.tight_layout()

plt.savefig(
    OUTPUT_PNG,
    dpi=200,
    bbox_inches="tight"
)

plt.show()


print("\nVisualisasi selesai.")
print(f"File: {OUTPUT_PNG}")