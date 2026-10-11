import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# ============================================================
# 1. NAMA FILE
# ============================================================

b04_file = "2026-10-06-00_00_2026-10-06-23_59_Sentinel-2_L2A_B04_(Raw).tiff"

b08_file = "2026-10-06-00_00_2026-10-06-23_59_Sentinel-2_L2A_B08_(Raw).tiff"

b11_file = "2026-10-06-00_00_2026-10-06-23_59_Sentinel-2_L2A_B11_(Raw).tiff"

hasil_file = "hasil_klasifikasi_jatim.tif"


# ============================================================
# 2. FUNGSI NORMALISASI
# ============================================================

def normalize(data):

    data = data.astype(float)

    # Hilangkan nilai NaN dan infinite
    data = np.nan_to_num(
        data,
        nan=0,
        posinf=0,
        neginf=0
    )

    # Ambil nilai persentil
    p2 = np.percentile(data, 2)
    p98 = np.percentile(data, 98)

    # Hindari pembagian dengan nol
    if p98 == p2:
        return np.zeros_like(data)

    # Normalisasi 0-1
    data = (data - p2) / (p98 - p2)

    # Batasi 0-1
    data = np.clip(data, 0, 1)

    return data


# ============================================================
# 3. BACA B04
# ============================================================

print("Membaca B04...")

with rasterio.open(b04_file) as src:

    B04 = src.read(1).astype(float)

    print("B04 berhasil dibaca")
    print("Ukuran B04:", B04.shape)


# ============================================================
# 4. BACA B08
# ============================================================

print("\nMembaca B08...")

with rasterio.open(b08_file) as src:

    B08 = src.read(1).astype(float)

    print("B08 berhasil dibaca")
    print("Ukuran B08:", B08.shape)


# ============================================================
# 5. BACA B11
# ============================================================

print("\nMembaca B11...")

with rasterio.open(b11_file) as src:

    B11 = src.read(1).astype(float)

    print("B11 berhasil dibaca")
    print("Ukuran B11:", B11.shape)


# ============================================================
# 6. CEK UKURAN BAND
# ============================================================

print("\n========================================")
print("CEK UKURAN BAND")
print("========================================")

print("B04 :", B04.shape)
print("B08 :", B08.shape)
print("B11 :", B11.shape)


# ============================================================
# 7. CEK APAKAH UKURANNYA SAMA
# ============================================================

if B04.shape != B08.shape or B04.shape != B11.shape:

    print("\nERROR:")
    print("Ukuran B04, B08, dan B11 berbeda.")

    print("\nUkuran:")
    print("B04:", B04.shape)
    print("B08:", B08.shape)
    print("B11:", B11.shape)

    print("\nProgram dihentikan.")

    raise SystemExit


print("\nSemua ukuran band sama.")


# ============================================================
# 8. BUAT FALSE COLOR COMPOSITE
# ============================================================
#
# Komposisi:
#
# Merah  = B11
# Hijau  = B08
# Biru   = B04
#
# Ini berguna untuk melihat vegetasi dan perbedaan
# antara area hutan/mangrove.
# ============================================================

print("\nMembuat False Color Composite...")

red = normalize(B11)
green = normalize(B08)
blue = normalize(B04)

rgb = np.dstack([
    red,
    green,
    blue
])

print("False Color Composite berhasil dibuat.")


# ============================================================
# 9. BACA HASIL KLASIFIKASI
# ============================================================

print("\nMembaca hasil klasifikasi...")

with rasterio.open(hasil_file) as src:

    hasil = src.read(1)

    print("Hasil klasifikasi berhasil dibaca.")
    print("Ukuran:", hasil.shape)


# ============================================================
# 10. CEK UKURAN HASIL KLASIFIKASI
# ============================================================

if hasil.shape != B04.shape:

    print("\nPERINGATAN:")
    print("Ukuran hasil klasifikasi berbeda dengan citra band.")

    print("B04:", B04.shape)
    print("Hasil:", hasil.shape)


# ============================================================
# 11. HITUNG JUMLAH PIXEL
# ============================================================

print("\n========================================")
print("JUMLAH PIXEL HASIL KLASIFIKASI")
print("========================================")

jumlah_building = np.sum(hasil == 1)
jumlah_farmland = np.sum(hasil == 2)
jumlah_water = np.sum(hasil == 3)
jumlah_forest = np.sum(hasil == 4)
jumlah_mangrove = np.sum(hasil == 5)

print(f"Building  : {jumlah_building:,}")
print(f"Farmland  : {jumlah_farmland:,}")
print(f"Water     : {jumlah_water:,}")
print(f"Forest    : {jumlah_forest:,}")
print(f"Mangrove  : {jumlah_mangrove:,}")


# ============================================================
# 12. COLORMAP HASIL KLASIFIKASI
# ============================================================

colors = [
    "white",     # 0 = NoData
    "red",       # 1 = Building
    "yellow",    # 2 = Farmland
    "blue",      # 3 = Water
    "green",     # 4 = Forest
    "purple"     # 5 = Mangrove
]

cmap = ListedColormap(colors)


# ============================================================
# 13. BUAT FIGURE
# ============================================================

print("\nMembuat visualisasi...")

fig, axes = plt.subplots(
    1,
    2,
    figsize=(18, 9)
)


# ============================================================
# 14. CITRA SENTINEL-2
# ============================================================

axes[0].imshow(rgb)

axes[0].set_title(
    "Citra Sentinel-2\nFalse Color B11-B08-B04",
    fontsize=15
)

axes[0].axis("off")


# ============================================================
# 15. HASIL KLASIFIKASI
# ============================================================

axes[1].imshow(
    hasil,
    cmap=cmap,
    vmin=0,
    vmax=5
)

axes[1].set_title(
    "Hasil Klasifikasi Random Forest",
    fontsize=15
)

axes[1].axis("off")


# ============================================================
# 16. LEGEND
# ============================================================

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
        facecolor="purple",
        label="Mangrove"
    )

]


fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=5,
    fontsize=12,
    title="Kelas Tutupan Lahan"
)


# ============================================================
# 17. JUDUL
# ============================================================

plt.suptitle(
    "Perbandingan Citra Sentinel-2 dan Hasil Klasifikasi\n"
    "Jawa Timur",
    fontsize=18
)


# ============================================================
# 18. ATUR LAYOUT
# ============================================================

plt.tight_layout(
    rect=[
        0,
        0.08,
        1,
        0.93
    ]
)


# ============================================================
# 19. SIMPAN GAMBAR
# ============================================================

output_file = "perbandingan_citra_klasifikasi.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 20. SELESAI
# ============================================================

print("\n========================================")
print("BERHASIL!")
print("========================================")

print("File hasil:")
print(output_file)

print("\nVisualisasi akan ditampilkan...")

plt.show()