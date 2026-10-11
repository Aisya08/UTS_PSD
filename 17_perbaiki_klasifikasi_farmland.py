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
from scipy.ndimage import label, uniform_filter

# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_TIF = os.path.join(BASE_DIR, "hasil_klasifikasi_jatim_FINAL.tif")
OUTPUT_TIF = os.path.join(BASE_DIR, "hasil_klasifikasi_jatim_FINAL.tif")
OUTPUT_PNG = os.path.join(BASE_DIR, "perbandingan_klasifikasi_farmland.png")

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


def perbaiki_persawahan():
    print("=" * 65)
    print("PENYEMPURNAAN AKURASI SAWAH / LAHAN PERTANIAN (FARMLAND)")
    print("=" * 65)

    # 1. Baca Raster Hasil Saat Ini
    print("\n[1/6] Membaca raster klasifikasi saat ini...")
    with rasterio.open(INPUT_TIF) as src:
        data = src.read(1)
        profile = src.profile.copy()
        crs = src.crs

    shape = data.shape
    print(f"Dimensi: {shape[1]} x {shape[0]} px | CRS: {crs}")

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
    print("\n[3/6] Menghitung indeks spektral (NDVI, NDBI, MNDWI)...")
    eps = 1e-6
    ndvi = (b08 - b04) / (b08 + b04 + eps)
    ndbi = (b11 - b08) / (b11 + b08 + eps)
    mndwi = (b03 - b11) / (b03 + b11 + eps)

    # 4. Analisis Kontekstual Spasial & Spektral Sawah
    print("\n[4/6] Mengidentifikasi anomali pada lanskap persawahan...")
    
    # Rasio keberadaan Farmland di sekitar jendela 7x7 (~70m radius)
    is_farm_initial = (data == FARMLAND).astype(np.float32)
    farm_ratio = uniform_filter(is_farm_initial, size=7)

    # A. Restorasi Padi Fase Vegetatif Rimbun (Bercak Forest di Dataran Sawah)
    # Patch Forest terisolasi berukuran kecil (<= 80 px) di zona dominan sawah
    lbl_forest, _ = label(data == FOREST, structure=np.ones((3, 3)))
    counts_forest = np.bincount(lbl_forest.ravel())
    small_forest_ids = np.where((counts_forest > 0) & (counts_forest <= 80))[0]
    forest_speckles = np.isin(lbl_forest, small_forest_ids) & (farm_ratio >= 0.35)
    print(f"  - Padi subur tertukar Hutan (bercak forest) terdeteksi: {np.sum(forest_speckles):,} px")

    # B. Restorasi Sawah Bera / Tanah Kering (Bercak Building di Tengah Sawah)
    # Patch Building kecil (<= 6 px) terisolasi di hamparan sawah dengan NDBI wajar tanah
    lbl_bldg, _ = label(data == BUILDING, structure=np.ones((3, 3)))
    counts_bldg = np.bincount(lbl_bldg.ravel())
    small_bldg_ids = np.where((counts_bldg > 0) & (counts_bldg <= 6))[0]
    bldg_speckles = np.isin(lbl_bldg, small_bldg_ids) & (farm_ratio >= 0.45) & (ndbi < 0.10)
    print(f"  - Sawah bera/kering tertukar Bangunan terdeteksi: {np.sum(bldg_speckles):,} px")

    # C. Restorasi Sawah Tergenang Air (Genangan dangkal di petak sawah)
    lbl_water, _ = label(data == WATER, structure=np.ones((3, 3)))
    counts_water = np.bincount(lbl_water.ravel())
    small_water_ids = np.where((counts_water > 0) & (counts_water <= 10))[0]
    water_speckles = np.isin(lbl_water, small_water_ids) & (farm_ratio >= 0.60)
    print(f"  - Sawah tergenang tertukar Perairan terdeteksi: {np.sum(water_speckles):,} px")

    # 5. Terapkan Reklasifikasi
    print("\n[5/6] Menerapkan perbaikan persawahan...")
    refined = data.copy()
    refined[forest_speckles] = FARMLAND
    refined[bldg_speckles] = FARMLAND
    refined[water_speckles] = FARMLAND

    # Pastikan integritas hutan pegunungan, laut, dan mangrove terlindungi 100%
    # Gugusan hutan besar (Gunung Ijen / Merapi / Raung & Alas Purwo) tetap Forest
    large_forest_ids = np.where(counts_forest >= 300)[0]
    large_forest_mask = np.isin(lbl_forest, large_forest_ids)
    refined[large_forest_mask & (data == FOREST)] = FOREST

    # Mangrove tetap Mangrove
    refined[data == MANGROVE] = MANGROVE

    # 6. Simpan Raster & Visualisasi
    print("\n[6/6] Menyimpan raster dan visualisasi komparasi...")
    profile.update(
        dtype="uint8",
        count=1,
        compress="lzw"
    )

    with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:
        dst.write(refined.astype(np.uint8), 1)
    print(f"Raster tersimpan: {OUTPUT_TIF}")

    # Visualisasi Komparasi (3 Panel)
    cmap = ListedColormap(COLORS)
    fig, axs = plt.subplots(1, 3, figsize=(21, 7), dpi=150)

    # Panel 1: Citra Satelit RGB
    rgb = np.stack([
        np.clip(b04 * 3.2, 0, 1),
        np.clip(b03 * 3.2, 0, 1),
        np.clip(b03 * 2.5, 0, 1)
    ], axis=-1)
    axs[0].imshow(rgb)
    axs[0].set_title("Citra Satelit Sentinel-2 L2A (RGB)", fontsize=13, fontweight="bold")
    axs[0].axis("off")

    # Panel 2: Klasifikasi Sebelum
    axs[1].imshow(data, cmap=cmap, vmin=1, vmax=5)
    axs[1].set_title("Klasifikasi Sebelum Perbaikan Sawah", fontsize=13, fontweight="bold")
    axs[1].axis("off")

    # Panel 3: Klasifikasi Sesudah
    axs[2].imshow(refined, cmap=cmap, vmin=1, vmax=5)
    axs[2].set_title("Klasifikasi Setelah Perbaikan Sawah (Kohesif)", fontsize=13, fontweight="bold")
    axs[2].axis("off")

    legend_patches = [
        Patch(facecolor=COLORS[0], label="1: Building (Perkotaan)"),
        Patch(facecolor=COLORS[1], label="2: Farmland (Sawah/Pertanian)"),
        Patch(facecolor=COLORS[2], label="3: Water (Perairan)"),
        Patch(facecolor=COLORS[3], label="4: Forest (Hutan)"),
        Patch(facecolor=COLORS[4], label="5: Mangrove (Mangrove)"),
    ]
    plt.legend(handles=legend_patches, loc="lower right", bbox_to_anchor=(1.0, -0.05), ncol=5, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Visualisasi perbandingan tersimpan: {OUTPUT_PNG}")

    # Cetak Statistik
    print("\n" + "=" * 65)
    print("STATISTIK TUTUPAN LAHAN SEBELUM VS SESUDAH PERBAIKAN SAWAH:")
    print("=" * 65)
    total_px = refined.size
    for code, name in CLASS_NAMES.items():
        count_old = np.sum(data == code)
        count_new = np.sum(refined == code)
        diff = count_new - count_old
        pct_new = (count_new / total_px) * 100.0
        sign = "+" if diff >= 0 else ""
        print(f"  {code}. {name:10}: {count_old:>8,} -> {count_new:>8,} ({sign}{diff:>7,}) [{pct_new:>5.2f}%]")
    print("=" * 65)


if __name__ == "__main__":
    perbaiki_persawahan()
