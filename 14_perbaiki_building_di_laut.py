import os
import glob
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import rasterio
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
from matplotlib.colors import ListedColormap, BoundaryNorm
# pyrefly: ignore [missing-import]
from matplotlib.patches import Patch
# pyrefly: ignore [missing-import]
from scipy.ndimage import label, binary_closing

# ============================================================
# 1. KONFIGURASI FILE & PARAMETER
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Sesuai ketentuan: Gunakan hasil_klasifikasi_jatim.tif sebagai raster sumber
INPUT_RASTER = os.path.join(BASE_DIR, "hasil_klasifikasi_jatim.tif")
OUTPUT_RASTER = os.path.join(BASE_DIR, "hasil_klasifikasi_jatim_building_diperbaiki.tif")

# File visualisasi hasil evaluasi
OUTPUT_COMPARISON_PNG = os.path.join(BASE_DIR, "perbandingan_building_di_laut.png")
OUTPUT_DIAGNOSTIC_PNG = os.path.join(BASE_DIR, "diagnostik_mask_laut.png")

# Definisi Kode Kelas
BUILDING = 1
FARMLAND = 2
WATER = 3
FOREST = 4
MANGROVE = 5

CLASS_NAMES = {
    BUILDING: "Building",
    FARMLAND: "Farmland",
    WATER: "Water",
    FOREST: "Forest",
    MANGROVE: "Mangrove",
}

# Skema warna resmi untuk visualisasi
COLORS = [
    "#FFFFFF",  # 0 = NoData
    "#E53935",  # 1 = Building (Merah)
    "#FDD835",  # 2 = Farmland (Kuning)
    "#1E88E5",  # 3 = Water (Biru)
    "#43A047",  # 4 = Forest (Hijau)
    "#8E24AA",  # 5 = Mangrove (Ungu)
]

CMAP = ListedColormap(COLORS)
NORM = BoundaryNorm(np.arange(-0.5, 6.5, 1), CMAP.N)


# ============================================================
# 2. FUNGSI BANTU
# ============================================================

def cari_band(nama_band):
    """Mencari file citra Sentinel-2 berdasarkan nama band."""
    pola = [
        os.path.join(BASE_DIR, f"*Sentinel-2_L2A_{nama_band}_(Raw).tiff"),
        os.path.join(BASE_DIR, f"*Sentinel-2_L2A_{nama_band}_(Raw).tif"),
    ]
    for pattern in pola:
        files = sorted(glob.glob(pattern))
        if files:
            return files[0]
    raise FileNotFoundError(
        f"File band {nama_band} tidak ditemukan di folder proyek: {BASE_DIR}"
    )


def baca_dan_validasi_band(nama_band, shape_target, crs_target, transform_target):
    """Membaca band Sentinel-2 dan memvalidasi kesesuaian CRS, transform, dan ukuran."""
    path = cari_band(nama_band)
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)

        if src.shape != shape_target:
            raise ValueError(
                f"Ukuran band {nama_band} ({src.shape}) tidak sesuai target ({shape_target})."
            )
        if src.crs != crs_target:
            raise ValueError(
                f"CRS band {nama_band} ({src.crs}) tidak sesuai target ({crs_target})."
            )
        if src.transform != transform_target:
            raise ValueError(
                f"Transform band {nama_band} tidak cocok dengan raster referensi."
            )

    print(f"  - {nama_band:4}: {os.path.basename(path)}")
    return data


def cari_laut_di_tepi(mask):
    """
    Mengidentifikasi komponen air yang terhubung ke salah satu batas tepi citra.
    Digunakan untuk memastikan badan air yang dideteksi adalah laut lepas/selat/pesisir.
    """
    struktur = np.ones((3, 3), dtype=np.uint8)
    labels, jumlah_komponen = label(mask.astype(bool), structure=struktur)

    # Ambil label komponen yang menyentuh batas tepi luar (atas, bawah, kiri, kanan)
    tepi = np.unique(
        np.concatenate([
            labels[0, :],
            labels[-1, :],
            labels[:, 0],
            labels[:, -1],
        ])
    )
    tepi = tepi[tepi > 0]

    if tepi.size == 0:
        return None, jumlah_komponen

    # Mask seluruh piksel yang tergabung dalam komponen tepi
    sea_mask = np.isin(labels, tepi)
    return sea_mask, jumlah_komponen


def normalisasi(data):
    """Normalisasi kontras 2-98 persen untuk visualisasi citra RGB/False Color."""
    valid = np.isfinite(data)
    if not np.any(valid):
        return np.zeros_like(data, dtype=np.float32)
    bawah, atas = np.percentile(data[valid], [2, 98])
    if atas <= bawah:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - bawah) / (atas - bawah), 0.0, 1.0)


def hitung_statistik_kelas(data):
    """Menghitung jumlah piksel untuk setiap kelas tutupan lahan."""
    return {
        CLASS_NAMES[code]: int(np.count_nonzero(data == code))
        for code in sorted(CLASS_NAMES.keys())
    }


# ============================================================
# 3. PROGRAM UTAMA
# ============================================================

def main():
    print("=" * 70)
    print("PERBAIKAN KELAS BUILDING DI WILAYAH LAUT")
    print("Program: 14_perbaiki_building_di_laut.py")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. BACA RASTER KLASIFIKASI ASLI
    # --------------------------------------------------------
    print("\n[1/6] Membaca raster klasifikasi asli...")
    if not os.path.exists(INPUT_RASTER):
        raise FileNotFoundError(f"File klasifikasi asli tidak ditemukan: {INPUT_RASTER}")

    with rasterio.open(INPUT_RASTER) as src:
        original = src.read(1)
        profile = src.profile.copy()
        crs = src.crs
        transform = src.transform
        shape = original.shape

    print(f"  Raster Sumber : {os.path.basename(INPUT_RASTER)}")
    print(f"  Dimensi       : {shape[1]} x {shape[0]} piksel")
    print(f"  Sistem CRS    : {crs}")

    stats_awal = hitung_statistik_kelas(original)
    bldg_awal = stats_awal["Building"]
    print(f"  Building Awal : {bldg_awal:,} piksel")

    # --------------------------------------------------------
    # 2. BACA BAND SENTINEL-2 (B03, B08, B11, dan B04 untuk False Color)
    # --------------------------------------------------------
    print("\n[2/6] Membaca dan memvalidasi band Sentinel-2 (B03, B04, B08, B11)...")
    b03 = baca_dan_validasi_band("B03", shape, crs, transform)
    b04 = baca_dan_validasi_band("B04", shape, crs, transform)
    b08 = baca_dan_validasi_band("B08", shape, crs, transform)
    b11 = baca_dan_validasi_band("B11", shape, crs, transform)

    # --------------------------------------------------------
    # 3. DETEKSI PERKIRAAN LAUT MENGGUNAKAN MNDWI & B08
    # --------------------------------------------------------
    print("\n[3/6] Menghitung MNDWI dan komponen air terhubung ke tepi citra...")
    penyebut = b03 + b11
    valid_water = (
        np.isfinite(b03)
        & np.isfinite(b08)
        & np.isfinite(b11)
        & (b03 >= 0)
        & (b08 >= 0)
        & (b11 >= 0)
        & (penyebut > 0)
    )

    mndwi = np.full(shape, np.nan, dtype=np.float32)
    np.divide(b03 - b11, penyebut, out=mndwi, where=valid_water)

    # Indeks vegetasi NDVI untuk membedakan daratan vs perairan dangkal/selat
    penyebut_ndvi = b08 + b04
    ndvi = np.full(shape, 0.0, dtype=np.float32)
    np.divide(b08 - b04, penyebut_ndvi, out=ndvi, where=penyebut_ndvi > 0)

    # Kriteria badan air laut & selat (mencakup laut dalam, pesisir, dan Selat Bali):
    # Air memiliki karakteristik penyerapan kuat di inframerah (B08 & B11 rendah)
    # serta NDVI rendah (< 0.15) dan MNDWI stabil
    kandidat_air = valid_water & (
        ((mndwi > 0.02) & (b08 <= 0.14))
        | ((mndwi > -0.05) & (b08 <= 0.135) & (b11 <= 0.135) & (ndvi <= 0.12))
    )

    # Ekstraksi komponen laut yang terhubung ke tepi citra (Selat Bali & Samudra Hindia)
    sea_mask, num_components = cari_laut_di_tepi(kandidat_air)

    # Evaluasi keabsahan deteksi laut
    if sea_mask is None or not np.any(sea_mask):
        print("\n[PERINGATAN] Deteksi laut tidak meyakinkan: tidak ada komponen air di tepi citra.")
        print("Perubahan TIDAK akan diterapkan secara otomatis.")
        print("Silakan periksa ambang MNDWI atau kualitas masking band Sentinel-2.")
        return

    # Tutup lubang kecil buih/gelombang di laut lepas agar mask laut utuh sempurna
    sea_mask = binary_closing(sea_mask, structure=np.ones((7, 7)))

    sea_pixels = int(np.count_nonzero(sea_mask))
    print(f"  Komponen Terdeteksi   : {num_components} komponen")
    print(f"  Luas Perkiraan Laut   : {sea_pixels:,} piksel ({sea_pixels / original.size * 100:.2f}% citra)")

    # --------------------------------------------------------
    # 4. IDENTIFIKASI & PERBAIKAN PIKSEL BUILDING DI WILAYAH LAUT
    # --------------------------------------------------------
    print("\n[4/6] Mengidentifikasi piksel Building di wilayah laut...")
    # Kondisi sesuai ketentuan:
    # 1. Kelas awalnya Building (kode 1)
    # 2. Berada di area laut yang terdeteksi
    target_building_in_sea = (original == BUILDING) & sea_mask
    jumlah_diubah = int(np.count_nonzero(target_building_in_sea))

    print(f"  Piksel Building di laut ditemukan : {jumlah_diubah:,} piksel")

    # Salin array asli agar raster asli tidak tertimpa
    hasil = original.copy()

    if jumlah_diubah > 0:
        # Ubah HANYA piksel tersebut menjadi Water (kode 3)
        hasil[target_building_in_sea] = WATER
        print(f"  Status: {jumlah_diubah} piksel berhasil diubah menjadi Water (kode 3).")
    else:
        print("  Status: Tidak ditemukan piksel Building di wilayah laut.")

    # --------------------------------------------------------
    # 5. VALIDASI & CETAK LAPORAN LENGKAP
    # --------------------------------------------------------
    print("\n[5/6] Verifikasi integritas data...")
    stats_akhir = hitung_statistik_kelas(hasil)
    bldg_sisa = stats_akhir["Building"]

    # Pastikan kelas selain piksel Building yang terdeteksi berada di laut tidak berubah
    assert stats_akhir["Farmland"] == stats_awal["Farmland"], "Peringatan: Kelas Farmland berubah!"
    assert stats_akhir["Forest"] == stats_awal["Forest"], "Peringatan: Kelas Forest berubah!"
    assert stats_akhir["Mangrove"] == stats_awal["Mangrove"], "Peringatan: Kelas Mangrove berubah!"
    assert bldg_sisa == bldg_awal - jumlah_diubah, "Peringatan: Selisih piksel Building tidak cocok!"
    assert stats_akhir["Water"] == stats_awal["Water"] + jumlah_diubah, "Peringatan: Selisih piksel Water tidak cocok!"

    print("  Integritas data terverifikasi 100% konsisten.")

    # Simpan hasil ke file baru
    profile.update(dtype="uint8", count=1, compress="lzw")
    with rasterio.open(OUTPUT_RASTER, "w", **profile) as dst:
        dst.write(hasil.astype(np.uint8), 1)
    print(f"  File raster tersimpan : {OUTPUT_RASTER}")

    # --------------------------------------------------------
    # 6. VISUALISASI PERBANDINGAN & DIAGNOSTIK
    # --------------------------------------------------------
    print("\n[6/6] Membuat visualisasi perbandingan dan diagnostik...")

    # A. Visualisasi Perbandingan 3 Panel
    false_color = np.dstack([
        normalisasi(b11),
        normalisasi(b08),
        normalisasi(b04),
    ])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=150)

    axes[0].imshow(false_color)
    axes[0].set_title("(a) Sentinel-2 False Color (B11-B08-B04)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(original, cmap=CMAP, norm=NORM, interpolation="nearest")
    axes[1].set_title("(b) Klasifikasi Asli (hasil_klasifikasi_jatim.tif)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(hasil, cmap=CMAP, norm=NORM, interpolation="nearest")
    axes[2].set_title("(c) Klasifikasi Setelah Perbaikan Building di Laut", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    legend_items = [
        Patch(facecolor=COLORS[1], label="1: Building"),
        Patch(facecolor=COLORS[2], label="2: Farmland"),
        Patch(facecolor=COLORS[3], label="3: Water"),
        Patch(facecolor=COLORS[4], label="4: Forest"),
        Patch(facecolor=COLORS[5], label="5: Mangrove"),
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=5,
        frameon=True,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.02)
    )

    fig.suptitle("Perbandingan Sebelum dan Sesudah Perbaikan Piksel Building di Laut", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(OUTPUT_COMPARISON_PNG, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  - Gambar perbandingan tersimpan : {OUTPUT_COMPARISON_PNG}")

    # B. Visualisasi Diagnostik Mask Laut Terpisah
    fig_diag, ax_diag = plt.subplots(1, 2, figsize=(14, 6.5), dpi=150)

    ax_diag[0].imshow(sea_mask, cmap="Blues", interpolation="nearest")
    ax_diag[0].set_title("Mask Perkiraan Laut Terdeteksi (Selat Bali & Samudra)", fontsize=11, fontweight="bold")
    ax_diag[0].axis("off")

    ax_diag[1].imshow(false_color)
    y_idx, x_idx = np.where(target_building_in_sea)
    if len(x_idx) > 0:
        ax_diag[1].scatter(
            x_idx,
            y_idx,
            color="#FF1744",
            s=10,
            marker="o",
            label=f"Building di Laut ({len(x_idx)} px)",
            edgecolors="yellow",
            linewidths=0.3
        )
        ax_diag[1].legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax_diag[1].set_title("Titik Lokasi Piksel Building yang Ditemukan di Wilayah Laut", fontsize=11, fontweight="bold")
    ax_diag[1].axis("off")

    fig_diag.suptitle("Diagnostik Mask Laut & Verifikasi Wilayah Perbaikan", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(OUTPUT_DIAGNOSTIC_PNG, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  - Gambar diagnostik tersimpan   : {OUTPUT_DIAGNOSTIC_PNG}")

    # --------------------------------------------------------
    # CETAK LAPORAN REKAPITULASI RESMI
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("LAPORAN PERBAIKAN PIKSEL BUILDING DI WILAYAH LAUT")
    print("=" * 70)
    print(f"  • Jumlah Piksel Building Awal         : {bldg_awal:>9,} piksel")
    print(f"  • Piksel Building Diubah Menjadi Water: {jumlah_diubah:>9,} piksel")
    print(f"  • Jumlah Piksel Building Tersisa      : {bldg_sisa:>9,} piksel")
    print("-" * 70)
    print(f"{'Kelas':<15} | {'Sebelum':>14} | {'Sesudah':>14} | {'Perubahan':>14}")
    print("-" * 70)

    for code in sorted(CLASS_NAMES.keys()):
        nama = CLASS_NAMES[code]
        awal = stats_awal[nama]
        akhir = stats_akhir[nama]
        selisih = akhir - awal
        tanda = f"+{selisih:,}" if selisih > 0 else f"{selisih:,}" if selisih < 0 else "0 (Tetap)"
        print(f"{code}. {nama:<12} | {awal:>14,} | {akhir:>14,} | {tanda:>14}")

    print("=" * 70)
    print("Proses selesai dengan sukses tanpa mengubah piksel daratan atau melatih ulang model.")
    print("=" * 70)


if __name__ == "__main__":
    main()
