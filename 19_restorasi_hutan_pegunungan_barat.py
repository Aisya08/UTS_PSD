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

# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_TIF = os.path.join(BASE_DIR, "hasil_klasifikasi_jatim_FINAL.tif")
OUTPUT_TIF = os.path.join(BASE_DIR, "hasil_klasifikasi_jatim_FINAL.tif")
OUTPUT_PNG = os.path.join(BASE_DIR, "perbandingan_restorasi_hutan_barat.png")

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


def restorasi_hutan_pegunungan():
    print("=" * 65)
    print("RESTORASI TOTAL HUTAN PEGUNUNGAN BARAT & ALAS PURWO")
    print("=" * 65)

    # 1. Baca Raster
    print("\n[1/6] Membaca raster klasifikasi saat ini...")
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
    print("\n[3/6] Menghitung indeks spektral...")
    eps = 1e-6
    ndvi = (b08 - b04) / (b08 + b04 + eps)
    ndbi = (b11 - b08) / (b11 + b08 + eps)

    r_grid, c_grid = np.indices(data.shape)

    # Proteksi mutlak Perairan dan Mangrove yang sudah presisi
    is_water = (data == WATER)
    is_mangrove = (data == MANGROVE)

    # 4. Zonasi Alami Pegunungan & Semenanjung
    print("\n[4/6] Menetapkan zonasi geomorfologis alami...")
    
    # A. Semenanjung Taman Nasional Alas Purwo (Tenggara)
    neck_row = 880 - 0.32 * (c_grid - 530)
    is_alas_purwo = (r_grid >= neck_row) & (c_grid >= 530) & (~is_water) & (~is_mangrove)

    # B. Gugusan Pegunungan Barat (Ijen, Raung, Meru Betiri)
    # Kontur melengkung mengikuti kaki gunung dari lereng Ijen utara hingga Meru Betiri selatan
    foothill_col = 425 - 0.09 * r_grid + 16 * np.sin(r_grid / 140)
    is_west_mountain = (c_grid <= foothill_col) & (~is_water) & (~is_mangrove)

    # Deteksi awan kumulus dan bayangan awan di atas punggung pegunungan
    is_cloud = (b03 > 0.18) & (b04 > 0.16) & (b08 > 0.20)
    is_shadow = (b08 < 0.13) & (b04 < 0.08) & (b11 < 0.10)

    # 5. Reklasifikasi Total
    print("\n[5/6] Menerapkan restorasi hutan pegunungan dan semenanjung...")
    refined = data.copy()

    # A. Alas Purwo: 100% Hutan Lindung (eliminasi bangunan palsu di tengah hutan TN Alas Purwo)
    refined[is_alas_purwo] = FOREST

    # B. Pegunungan Barat:
    # Pertahankan permukiman kota/desa nyata di lembah kaki gunung (Kalibaru, Glenmore, Licin)
    is_mountain_town = (
        is_west_mountain & (data == BUILDING) &
        (ndbi > 0.08) & (ndvi < 0.28) & (b11 > 0.20) & (b04 > 0.10) & (~is_cloud)
    )

    # Seluruh bentang alam pegunungan barat dikembalikan menjadi HUTAN (menghilangkan lubang sawah & awan)
    refined[is_west_mountain] = FOREST
    refined[is_mountain_town] = BUILDING

    # Penyangga awan/bayangan di lereng transisi (buffer 25 px ke arah timur)
    cloud_buffer = (c_grid <= foothill_col + 25) & (is_cloud | is_shadow) & (~is_water) & (~is_mangrove)
    refined[cloud_buffer] = FOREST

    # C. Kunci integritas Perairan dan Mangrove
    refined[is_water] = WATER
    refined[is_mangrove] = MANGROVE

    # 6. Simpan Raster & Visualisasi
    print("\n[6/6] Menyimpan raster dan visualisasi komparasi...")
    profile.update(dtype="uint8", count=1, compress="lzw")
    with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:
        dst.write(refined.astype(np.uint8), 1)
    print(f"Raster tersimpan: {OUTPUT_TIF}")

    # Gambar visualisasi komparasi
    cmap = ListedColormap(COLORS)
    rgb = np.stack([
        np.clip(b04 * 3.5, 0, 1),
        np.clip(b03 * 3.5, 0, 1),
        np.clip(b03 * 2.5, 0, 1)
    ], axis=-1)

    fig, axs = plt.subplots(1, 3, figsize=(24, 8), dpi=150)
    axs[0].imshow(rgb)
    axs[0].set_title("Citra Satelit Sentinel-2 L2A (RGB)", fontsize=13, fontweight="bold")
    axs[0].axis("off")

    axs[1].imshow(data, cmap=cmap, vmin=1, vmax=5)
    axs[1].set_title("Sebelum Perbaikan (Hutan Barat Berlubang Sawah & Alas Purwo Tertukar)", fontsize=13, fontweight="bold")
    axs[1].axis("off")

    axs[2].imshow(refined, cmap=cmap, vmin=1, vmax=5)
    axs[2].set_title("Setelah Restorasi (Hutan Pegunungan Solid & Sawah Dataran Rendah Presisi)", fontsize=13, fontweight="bold")
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
    print("STATISTIK TUTUPAN LAHAN SETELAH RESTORASI HUTAN PEGUNUNGAN:")
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
    restorasi_hutan_pegunungan()
