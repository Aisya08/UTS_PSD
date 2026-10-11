import os
import glob
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import rasterio
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
from matplotlib.colors import ListedColormap
# pyrefly: ignore [missing-import]
from matplotlib.patches import Patch
# pyrefly: ignore [missing-import]
from scipy.ndimage import label, binary_closing

# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_TIF = os.path.join(BASE_DIR, "hasil_klasifikasi_jatim_mangrove_koridor_pesisir.tif")
OUTPUT_TIF = os.path.join(BASE_DIR, "hasil_klasifikasi_jatim_FINAL.tif")
OUTPUT_PNG = os.path.join(BASE_DIR, "hasil_klasifikasi_jatim_FINAL.png")

BUILDING = 1
FARMLAND = 2
WATER = 3
FOREST = 4
MANGROVE = 5

CLASS_NAMES = {
    1: "Building",
    2: "Farmland",
    3: "Water",
    4: "Forest",
    5: "Mangrove"
}

# Warna visualisasi resmi
COLORS = [
    "#E53935",  # 1 = Building (Merah)
    "#FBC02D",  # 2 = Farmland (Kuning)
    "#1E88E5",  # 3 = Water (Biru)
    "#2E7D32",  # 4 = Forest (Hijau)
    "#8E24AA",  # 5 = Mangrove (Ungu)
]


def cari_band(nama):
    patterns = [
        os.path.join(BASE_DIR, f"*Sentinel-2_L2A_{nama}_(Raw).tiff"),
        os.path.join(BASE_DIR, f"*Sentinel-2_L2A_{nama}_(Raw).tif"),
    ]
    for pattern in patterns:
        files = sorted(glob.glob(pattern))
        if files:
            return files[0]
    raise FileNotFoundError(f"Band {nama} tidak ditemukan.")


def perbaiki_klasifikasi():
    print("=" * 65)
    print("PERBAIKAN AKURASI PERKOTAAN & ELIMINASI PERAIRAN PALSU (WATER)")
    print("=" * 65)

    # 1. Baca Raster Hasil Final Sebelumnya
    print("\n[1/6] Membaca raster klasifikasi saat ini...")
    with rasterio.open(INPUT_TIF) as src:
        data = src.read(1)
        profile = src.profile.copy()
        crs = src.crs

    shape = data.shape
    print(f"Dimensi: {shape[1]} x {shape[0]} px")
    print(f"CRS: {crs}")

    # 2. Baca Band Sentinel-2
    print("\n[2/6] Membaca kanal spektral Sentinel-2...")
    with rasterio.open(cari_band("B03")) as src:
        b03 = src.read(1).astype(np.float32)
    with rasterio.open(cari_band("B04")) as src:
        b04 = src.read(1).astype(np.float32)
    with rasterio.open(cari_band("B08")) as src:
        b08 = src.read(1).astype(np.float32)
    with rasterio.open(cari_band("B11")) as src:
        b11 = src.read(1).astype(np.float32)

    # 3. Hitung Indeks Spektral
    print("\n[3/6] Menghitung indeks spektral (MNDWI, NDWI, NDVI, NDBI)...")
    eps = 1e-6
    mndwi = (b03 - b11) / (b03 + b11 + eps)
    ndwi = (b03 - b08) / (b03 + b08 + eps)
    ndvi = (b08 - b04) / (b08 + b04 + eps)
    ndbi = (b11 - b08) / (b11 + b08 + eps)

    # 4. Deteksi Badan Air Sebenarnya (Laut & Perairan Nyata)
    print("\n[4/6] Memisahkan perairan nyata vs perairan palsu di darat/kota...")
    
    # Deteksi air laut di tepi citra (mencakup laut lepas, pesisir, dan Selat Bali)
    penyebut_ndvi = b08 + b04
    ndvi_calc = np.full(shape, 0.0, dtype=np.float32)
    np.divide(b08 - b04, penyebut_ndvi, out=ndvi_calc, where=penyebut_ndvi > 0)

    candidates_sea = (
        ((mndwi > 0.02) & (b08 <= 0.14))
        | ((mndwi > -0.05) & (b08 <= 0.135) & (b11 <= 0.135) & (ndvi_calc <= 0.12))
    )
    labels, _ = label(candidates_sea, structure=np.ones((3, 3)))
    tepi = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
    tepi = tepi[tepi > 0]
    sea_mask = np.isin(labels, tepi)

    # Badan air daratan nyata (sungai/danau/waduk) memiliki absorbsi kuat (MNDWI > 0.05, NIR rendah, bukan vegetasi/bangunan)
    is_true_inland_water = (
        (data == WATER)
        & (mndwi > 0.05)
        & (b08 <= 0.14)
        & (b11 <= 0.14)
        & (ndvi < 0.25)
        & (ndbi < 0.0)
    )
    is_true_water = sea_mask | is_true_inland_water

    # False Water: Piksel yang terlanjur terlabel Water padahal berada di daratan perkotaan/pertanian
    is_false_water = (data == WATER) & (~is_true_water)
    print(f"  - Perairan palsu (bayangan kota/aspal/sawah) terdeteksi: {np.sum(is_false_water):,} px")

    # Alokasi false water ke kelas aslinya
    to_bldg_from_water = is_false_water & ((ndbi > -0.02) | ((b11 > 0.15) & (b04 > 0.08))) & (ndvi < 0.35)
    to_forest_from_water = is_false_water & (ndvi > 0.55) & (~to_bldg_from_water)
    to_farm_from_water = is_false_water & (~to_bldg_from_water) & (~to_forest_from_water)

    print(f"    * Dialihkan ke Perkotaan (Bangunan/Jalan): {np.sum(to_bldg_from_water):,} px")
    print(f"    * Dialihkan ke Hutan: {np.sum(to_forest_from_water):,} px")
    print(f"    * Dialihkan ke Sawah/Pertanian: {np.sum(to_farm_from_water):,} px")

    # 5. Terapkan Reklasifikasi Komprehensif
    print("\n[5/6] Menerapkan perbaikan perkotaan dan perairan...")
    refined = data.copy()

    # Perbaiki perairan palsu
    refined[to_farm_from_water] = FARMLAND
    refined[to_bldg_from_water] = BUILDING
    refined[to_forest_from_water] = FOREST
    refined[is_true_water] = WATER

    # Eliminasi false building pada vegetasi lebat (NDVI tinggi)
    false_bldg = (refined == BUILDING) & ((ndvi > 0.36) | (ndbi < -0.08))
    false_bldg_forest = false_bldg & (ndvi > 0.55)
    refined[false_bldg] = FARMLAND
    refined[false_bldg_forest] = FOREST
    print(f"  - False building pada vegetasi lebat dibersihkan: {np.sum(false_bldg):,} px")

    # Deteksi true building pada Farmland (permukiman, ruko, perumahan nyata)
    true_bldg = (refined == FARMLAND) & (ndbi > 0.05) & (ndvi < 0.30) & (b11 > 0.18) & (b04 > 0.08)
    refined[true_bldg] = BUILDING
    print(f"  - Permukiman nyata dari lahan pertanian ditambahkan: {np.sum(true_bldg):,} px")

    # Pastikan integritas Mangrove & Perairan sejati tidak terganggu
    refined[data == MANGROVE] = MANGROVE
    sea_closed = binary_closing(sea_mask, structure=np.ones((7, 7)))
    refined[sea_closed] = WATER
    refined[is_true_water] = WATER

    # 6. Simpan Hasil
    print("\n[6/6] Menyimpan raster dan visualisasi...")
    profile.update(
        dtype="uint8",
        count=1,
        compress="lzw"
    )

    with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:
        dst.write(refined.astype(np.uint8), 1)
    print(f"Raster tersimpan: {OUTPUT_TIF}")

    # Visualisasi
    cmap = ListedColormap(COLORS)
    plt.figure(figsize=(10, 8), dpi=150)
    plt.imshow(refined, cmap=cmap, vmin=1, vmax=5)
    plt.title("Hasil Klasifikasi Tutupan Lahan Jawa Timur (Akurasi Perkotaan & Perairan)", fontsize=13, fontweight="bold")
    plt.axis("off")

    legend_patches = [
        Patch(facecolor=COLORS[0], label="1: Building (Perkotaan)"),
        Patch(facecolor=COLORS[1], label="2: Farmland (Pertanian)"),
        Patch(facecolor=COLORS[2], label="3: Water (Perairan)"),
        Patch(facecolor=COLORS[3], label="4: Forest (Hutan)"),
        Patch(facecolor=COLORS[4], label="5: Mangrove (Mangrove)"),
    ]
    plt.legend(handles=legend_patches, loc="lower right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Gambar tersimpan: {OUTPUT_PNG}")

    print("\n" + "=" * 65)
    print("STATISTIK TUTUPAN LAHAN SETELAH PERBAIKAN PERKOTAAN & PERAIRAN:")
    print("=" * 65)
    total_px = refined.size
    for code, name in CLASS_NAMES.items():
        count = np.sum(refined == code)
        pct = (count / total_px) * 100.0
        print(f"  {code}. {name:10}: {count:>8,} px ({pct:>6.2f}%)")
    print("=" * 65)


if __name__ == "__main__":
    perbaiki_klasifikasi()
