import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# ==============================
# FILE INPUT
# ==============================
input_file = "hasil_klasifikasi_jatim.tif"

# ==============================
# BACA RASTER
# ==============================
with rasterio.open(input_file) as src:
    data = src.read(1)

print("Raster berhasil dibaca")
print("Ukuran raster:", data.shape)

# ==============================
# HITUNG JUMLAH PIXEL
# ==============================
classes = {
    1: "Building",
    2: "Farmland",
    3: "Water",
    4: "Forest",
    5: "Mangrove"
}

print("\nJumlah pixel tiap kelas:")

for value, name in classes.items():
    jumlah = (data == value).sum()
    print(f"{name:10s}: {jumlah:,}")

# ==============================
# WARNA KELAS
# ==============================
# 0 = NoData
# 1 = Building
# 2 = Farmland
# 3 = Water
# 4 = Forest
# 5 = Mangrove

colors = [
    "white",      # NoData
    "red",        # Building
    "yellow",     # Farmland
    "blue",       # Water
    "green",      # Forest
    "purple"      # Mangrove
]

cmap = ListedColormap(colors)

# ==============================
# TAMPILKAN HASIL
# ==============================
plt.figure(figsize=(12, 9))

plt.imshow(
    data,
    cmap=cmap,
    vmin=0,
    vmax=5
)

plt.title(
    "Hasil Klasifikasi Tutupan Lahan\n"
    "Jawa Timur - Random Forest",
    fontsize=16
)

plt.axis("off")

# ==============================
# LEGEND
# ==============================
legend_elements = [
    Patch(facecolor="red", label="Building"),
    Patch(facecolor="yellow", label="Farmland"),
    Patch(facecolor="blue", label="Water"),
    Patch(facecolor="green", label="Forest"),
    Patch(facecolor="purple", label="Mangrove")
]

plt.legend(
    handles=legend_elements,
    loc="lower right",
    title="Kelas Tutupan Lahan"
)

# ==============================
# SIMPAN GAMBAR
# ==============================
output_file = "hasil_klasifikasi_jatim.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

print("\nGambar berhasil disimpan:")
print(output_file)

plt.show()