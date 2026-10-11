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
from scipy.ndimage import gaussian_filter

# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_TIF = os.path.join(BASE_DIR, "hasil_klasifikasi_jatim_FINAL.tif")
OUTPUT_TIF = os.path.join(BASE_DIR, "hasil_klasifikasi_jatim_FINAL.tif")
OUTPUT_PNG = os.path.join(BASE_DIR, "perbandingan_harmonisasi_sawah_hutan.png")

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


def harmonisasi_sawah_hutan():
    print("=" * 65)
    print("HARMONISASI SPASIAL-SPEKTRAL SAWAH VS HUTAN (ZONASI ALAMI)")
    print("=" * 65)

    # 1. Baca Raster
    print("\n[1/6] Membaca raster klasifikasi...")
    with rasterio.open(INPUT_TIF) as src:
        data = src.read(1)
        profile = src.profile.copy()
        crs = src.crs

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
    print("\n[3/6] Menghitung indeks spektral (NDVI, NDBI, NDMI)...")
    eps = 1e-6
    ndvi = (b08 - b04) / (b08 + b04 + eps)
    ndbi = (b11 - b08) / (b11 + b08 + eps)
    ndmi = (b08 - b11) / (b08 + b11 + eps)

    # 4. Zonasi Alami & Medan Densitas Kontur
    print("\n[4/6] Membangun medan densitas spasial kontur alami...")
    rows, cols = np.indices(data.shape)
    water_mask = (data == WATER)
    mangrove_mask = (data == MANGROVE)

    # Garis leher alami Semenanjung Alas Purwo antara Teluk Grajagan dan Teluk Pangpang
    neck_row = 880 - 0.32 * (cols - 530)
    is_peninsula = (rows >= neck_row) & (cols >= 530) & (~water_mask) & (~mangrove_mask)

    # Biji Hutan Berkepercayaan Tinggi (High-confidence Forest Seeds)
    forest_seed = (
        ((ndvi > 0.58) & (b04 < 0.085) & (ndmi > 0.10) & (~water_mask) & (~mangrove_mask)) |
        is_peninsula
    ).astype(np.float32)

    # Biji Sawah/Pertanian (High-confidence Farmland Seeds)
    farm_seed = (
        (data == FARMLAND) & (b04 > 0.09) & (b11 > 0.22) & (~is_peninsula) & (~water_mask) & (~mangrove_mask)
    ).astype(np.float32)

    # Filter Gaussian untuk kontur alami pegunungan vs dataran sawah (~200m radius)
    smooth_forest = gaussian_filter(forest_seed, sigma=10)
    smooth_farm = gaussian_filter(farm_seed, sigma=10)
    forest_zone = (smooth_forest > smooth_farm * 0.82) & (~water_mask) & (~mangrove_mask)

    # 5. Reklasifikasi Harmonis
    print("\n[5/6] Menerapkan reklasifikasi harmonis...")
    refined = data.copy()

    # Zona Hutan (Pegunungan Ijen-Raung, Meru Betiri, Alas Purwo) menjadi Hutan solid
    refined[forest_zone] = FOREST

    # Zona Sawah (Dataran rendah Banyuwangi) menjadi Sawah kohesif tanpa bercak hutan acak
    refined[(~forest_zone) & (refined == FOREST)] = FARMLAND

    # Proteksi Permukiman Nyata (Building): Pusat kota, jalan arteri, desa
    is_real_bldg = (data == BUILDING) & (
        (ndbi > -0.02) | ((b11 > 0.18) & (b04 > 0.09))
    ) & (ndvi < 0.35) & (~is_peninsula)
    refined[is_real_bldg] = BUILDING

    # Proteksi Mutlak Perairan dan Mangrove
    refined[water_mask] = WATER
    refined[mangrove_mask] = MANGROVE

    # 6. Simpan Raster & Visualisasi
    print("\n[6/6] Menyimpan raster dan visualisasi komparasi...")
    profile.update(dtype="uint8", count=1, compress="lzw")
    with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:
        dst.write(refined.astype(np.uint8), 1)
    print(f"Raster tersimpan: {OUTPUT_TIF}")

    # Visualisasi
    cmap = ListedColormap(COLORS)
    fig, axs = plt.subplots(1, 3, figsize=(21, 7), dpi=150)

    # RGB
    rgb = np.stack([
        np.clip(b04 * 3.2, 0, 1),
        np.clip(b03 * 3.2, 0, 1),
        np.clip(b03 * 2.5, 0, 1)
    ], axis=-1)
    axs[0].imshow(rgb)
    axs[0].set_title("Citra Satelit Sentinel-2 L2A (RGB Asli)", fontsize=13, fontweight="bold")
    axs[0].axis("off")

    # Sebelum
    axs[1].imshow(data, cmap=cmap, vmin=1, vmax=5)
    axs[1].set_title("Sebelum Harmonisasi (Sawah & Hutan Bercampur)", fontsize=13, fontweight="bold")
    axs[1].axis("off")

    # Sesudah
    axs[2].imshow(refined, cmap=cmap, vmin=1, vmax=5)
    axs[2].set_title("Setelah Harmonisasi (Zonasi Alami Sawah & Hutan)", fontsize=13, fontweight="bold")
    axs[2].axis("off")

    legend_patches = [
        Patch(facecolor=COLORS[0], label="1: Building (Perkotaan)"),
        Patch(facecolor=COLORS[1], label="2: Farmland (Sawah/Pertanian)"),
        Patch(facecolor=COLORS[2], label="3: Water (Perairan)"),
        Patch(facecolor=COLORS[4], label="4: Forest (Hutan)"),
        Patch(facecolor=COLORS[4], label="5: Mangrove (Mangrove)"),
    ]
    plt.legend(handles=legend_patches, loc="lower right", bbox_to_anchor=(1.0, -0.05), ncol=5, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Visualisasi perbandingan tersimpan: {OUTPUT_PNG}")

    # Cetak Statistik
    print("\n" + "=" * 65)
    print("STATISTIK TUTUPAN LAHAN SETELAH HARMONISASI SAWAH & HUTAN:")
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
    harmonisasi_sawah_hutan()
