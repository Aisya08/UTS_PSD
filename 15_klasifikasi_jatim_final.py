import numpy as np
import pandas as pd
import rasterio
import joblib
import matplotlib.pyplot as plt

from glob import glob
from scipy.ndimage import label, distance_transform_edt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


# ============================================================
# KONFIGURASI FINAL
# ============================================================

INPUT_CLASSIFICATION = "hasil_klasifikasi_jatim.tif"
MODEL_FILE = "random_forest_jatim.pkl"

OUTPUT_TIF = "hasil_klasifikasi_jatim_FINAL.tif"
OUTPUT_PNG = "hasil_klasifikasi_jatim_FINAL.png"

FEATURES = ["B04", "B08", "B8A", "B11"]

CLASS_CODES = {
    "Building": 1,
    "Farmland": 2,
    "Water": 3,
    "Forest": 4,
    "Mangrove": 5,
}

BUILDING = 1
FARMLAND = 2
WATER = 3
FOREST = 4
MANGROVE = 5

# Parameter awal filter pesisir
MNDWI_THRESHOLD = 0.05
WATER_B08_PERCENTILE = 90
COASTAL_BUFFER_M = 1000

# Mangrove dengan nilai MNDWI di atas batas ini
# tidak akan dipertahankan sebagai Mangrove.
MANGROVE_MNDWI_MAX = 0.20

BATCH_SIZE = 50000


# ============================================================
# FUNGSI BANTU
# ============================================================

def cari_band(nama):
    patterns = [
        f"*Sentinel-2_L2A_{nama}_(Raw).tiff",
        f"*Sentinel-2_L2A_{nama}_(Raw).tif",
    ]

    for pattern in patterns:
        files = sorted(glob(pattern))

        if files:
            return files[0]

    raise FileNotFoundError(
        f"Band {nama} tidak ditemukan di folder proyek."
    )


def baca_band(nama, shape, crs, transform):
    path = cari_band(nama)

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)

        if src.shape != shape:
            raise ValueError(
                f"Ukuran band {nama} tidak sesuai."
            )

        if src.crs != crs or src.transform != transform:
            raise ValueError(
                f"CRS atau transform band {nama} tidak sesuai."
            )

    print(f"{nama}: {path}")
    return data


def cari_laut_di_tepi(mask):
    """
    Memilih semua komponen kandidat air yang menyentuh
    salah satu batas citra.
    """
    labels, jumlah_komponen = label(
        mask.astype(bool),
        structure=np.ones((3, 3), dtype=np.uint8),
    )

    tepi = np.unique(
        np.concatenate([
            labels[0, :],
            labels[-1, :],
            labels[:, 0],
            labels[:, -1],
        ])
    )

    tepi = tepi[tepi > 0]

    if len(tepi) == 0:
        return None, jumlah_komponen

    return np.isin(labels, tepi), jumlah_komponen


def normalisasi(data):
    valid = np.isfinite(data)

    if not np.any(valid):
        return np.zeros_like(data, dtype=np.float32)

    nilai = data[valid]
    bawah, atas = np.percentile(nilai, [2, 98])

    if atas <= bawah:
        return np.zeros_like(data, dtype=np.float32)

    return np.clip(
        (data - bawah) / (atas - bawah),
        0,
        1,
    )


def hitung_kelas(data):
    return {
        nama: int(np.count_nonzero(data == kode))
        for nama, kode in CLASS_CODES.items()
    }


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():

    print("=" * 65)
    print("KLASIFIKASI JAWA TIMUR - HASIL FINAL")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. BACA KLASIFIKASI AWAL
    # --------------------------------------------------------

    print("\n[1/7] Membaca klasifikasi awal...")

    with rasterio.open(INPUT_CLASSIFICATION) as src:
        original = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        bounds = src.bounds
        crs = src.crs

    if crs is None or crs.to_epsg() != 4326:
        raise ValueError(
            "Raster harus menggunakan EPSG:4326."
        )

    shape = original.shape

    print("Ukuran raster:", shape)
    print("CRS:", crs)

    # --------------------------------------------------------
    # 2. BACA BAND SATELIT
    # --------------------------------------------------------

    print("\n[2/7] Membaca band Sentinel-2...")

    bands = {}

    for name in ["B03", "B04", "B08", "B8A", "B11"]:
        bands[name] = baca_band(
            name,
            shape,
            crs,
            transform,
        )

    b03 = bands["B03"]
    b04 = bands["B04"]
    b08 = bands["B08"]
    b8a = bands["B8A"]
    b11 = bands["B11"]

    # --------------------------------------------------------
    # 3. DETEKSI KANDIDAT LAUT MENGGUNAKAN MNDWI
    # --------------------------------------------------------

    print("\n[3/7] Menghitung MNDWI dan kandidat air...")

    denominator = b03 + b11

    valid_water = (
        np.isfinite(b03)
        & np.isfinite(b08)
        & np.isfinite(b11)
        & (b03 >= 0)
        & (b08 >= 0)
        & (b11 >= 0)
        & (denominator > 0)
    )

    mndwi = np.full(shape, np.nan, dtype=np.float32)

    np.divide(
        b03 - b11,
        denominator,
        out=mndwi,
        where=valid_water,
    )

    # Nilai B08 kelas Water digunakan sebagai acuan spektral.
    water_reference = (
        (original == WATER)
        & valid_water
    )

    if not np.any(water_reference):
        raise RuntimeError(
            "Kelas Water tidak memiliki pixel valid."
        )

    b08_limit = float(
        np.percentile(
            b08[water_reference],
            WATER_B08_PERCENTILE,
        )
    )

    candidates = (
        valid_water
        & (mndwi > MNDWI_THRESHOLD)
        & (b08 <= b08_limit)
    )

    sea_mask, num_components = cari_laut_di_tepi(
        candidates
    )

    if sea_mask is None or not np.any(sea_mask):
        raise RuntimeError(
            "Deteksi laut gagal. "
            "Tidak ada kandidat air yang menyentuh tepi citra."
        )

    # Hanya komponen air yang bersambung ke tepi citra
    # yang dipakai sebagai perkiraan laut.
    sea_pixels = int(np.count_nonzero(sea_mask))

    print("Ambang MNDWI:", MNDWI_THRESHOLD)
    print("Batas B08:", round(b08_limit, 5))
    print("Komponen kandidat:", num_components)
    print("Pixel perkiraan laut:", f"{sea_pixels:,}")

    # --------------------------------------------------------
    # 4. BENTUK KORIDOR PESISIR
    # --------------------------------------------------------

    print("\n[4/7] Membentuk koridor pesisir...")

    latitude = (bounds.top + bounds.bottom) / 2

    pixel_y_m = abs(transform.e) * 111320.0
    pixel_x_m = (
        abs(transform.a)
        * 111320.0
        * np.cos(np.deg2rad(latitude))
    )

    distance_to_sea = distance_transform_edt(
        ~sea_mask,
        sampling=(pixel_y_m, pixel_x_m),
    )

    coastal_zone = (
        distance_to_sea <= COASTAL_BUFFER_M
    )

    # Pertahankan Mangrove hanya jika:
    # - berada dalam koridor pesisir;
    # - bukan piksel yang terdeteksi sebagai laut;
    # - nilai MNDWI tidak terlalu menyerupai air terbuka.
    valid_all_bands = (
        np.isfinite(b04)
        & np.isfinite(b08)
        & np.isfinite(b8a)
        & np.isfinite(b11)
        & (b04 >= 0)
        & (b08 >= 0)
        & (b8a >= 0)
        & (b11 >= 0)
    )

    keep_mangrove = (
        (original == MANGROVE)
        & coastal_zone
        & (~sea_mask)
        & valid_all_bands
        & np.isfinite(mndwi)
        & (mndwi <= MANGROVE_MNDWI_MAX)
    )

    change_mangrove = (
        (original == MANGROVE)
        & (~keep_mangrove)
    )

    print("Batas koridor:", COASTAL_BUFFER_M, "meter")
    print(
        "Mangrove dipertahankan:",
        f"{np.count_nonzero(keep_mangrove):,}"
    )
    print(
        "Mangrove yang direklasifikasi:",
        f"{np.count_nonzero(change_mangrove):,}"
    )

    # --------------------------------------------------------
    # 5. REKLASIFIKASI MANGROVE DI LUAR KORIDOR
    # --------------------------------------------------------

    print("\n[5/7] Mereklasifikasi piksel Mangrove...")

    model = joblib.load(MODEL_FILE)

    hasil = original.copy()

    model_classes = [str(x) for x in model.classes_]

    alternate_columns = [
        i
        for i, class_name in enumerate(model_classes)
        if class_name != "Mangrove"
    ]

    if not alternate_columns:
        raise RuntimeError(
            "Model tidak memiliki kelas alternatif."
        )

    model_class_codes = np.array([
        CLASS_CODES[class_name]
        for class_name in model_classes
    ], dtype=np.uint8)

    pixel_indices = np.flatnonzero(
        change_mangrove.ravel()
    )

    total_batches = (
        len(pixel_indices) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    for batch_number, start in enumerate(
        range(0, len(pixel_indices), BATCH_SIZE),
        start=1,
    ):
        idx = pixel_indices[
            start:start + BATCH_SIZE
        ]

        X = pd.DataFrame({
            name: bands[name].ravel()[idx]
            for name in FEATURES
        })

        probabilities = model.predict_proba(X)

        alternate_probabilities = probabilities[
            :, alternate_columns
        ]

        best_positions = np.argmax(
            alternate_probabilities,
            axis=1,
        )

        chosen_columns = np.asarray(
            alternate_columns
        )[best_positions]

        chosen_codes = model_class_codes[
            chosen_columns
        ]

        hasil.ravel()[idx] = chosen_codes

        print(
            f"Batch {batch_number}/{total_batches} selesai."
        )

    # Pastikan Mangrove pesisir yang dipertahankan
    # tidak berubah setelah reklasifikasi.
    hasil[keep_mangrove] = MANGROVE

    # Mask laut yang terdeteksi menjadi Water.
    # Ini mencegah Mangrove tetap berwarna ungu
    # pada area laut yang masuk mask.
    hasil[sea_mask] = WATER

    # --------------------------------------------------------
    # 6. SIMPAN RASTER FINAL
    # --------------------------------------------------------

    print("\n[6/7] Menyimpan hasil final...")

    profile.update(
        dtype="uint8",
        count=1,
        compress="lzw",
    )

    with rasterio.open(
        OUTPUT_TIF,
        "w",
        **profile,
    ) as dst:
        dst.write(hasil.astype(np.uint8), 1)

    print("Raster final:", OUTPUT_TIF)

    # --------------------------------------------------------
    # 7. BUAT GAMBAR FINAL
    # --------------------------------------------------------

    print("\n[7/7] Membuat visualisasi final...")

    false_color = np.dstack([
        normalisasi(b11),
        normalisasi(b08),
        normalisasi(b04),
    ])

    colors = [
        "#FFFFFF",
        "#E53935",
        "#FDD835",
        "#1E88E5",
        "#43A047",
        "#8E24AA",
    ]

    cmap = ListedColormap(colors)

    norm = BoundaryNorm(
        np.arange(-0.5, 6.5, 1),
        cmap.N,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 7),
    )

    axes[0].imshow(false_color)
    axes[0].set_title("Sentinel-2 False Color")
    axes[0].axis("off")

    axes[1].imshow(
        original,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    axes[1].set_title("Klasifikasi Awal")
    axes[1].axis("off")

    axes[2].imshow(
        hasil,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    axes[2].set_title("Klasifikasi Final")
    axes[2].axis("off")

    legend_items = [
        Patch(facecolor=colors[1], label="Building"),
        Patch(facecolor=colors[2], label="Farmland"),
        Patch(facecolor=colors[3], label="Water"),
        Patch(facecolor=colors[4], label="Forest"),
        Patch(facecolor=colors[5], label="Mangrove"),
    ]

    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=5,
        frameon=False,
    )

    fig.suptitle(
        "Klasifikasi Tutupan Lahan Menggunakan Sentinel-2",
        fontsize=15,
    )

    plt.tight_layout(
        rect=[0, 0.09, 1, 0.93]
    )

    plt.savefig(
        OUTPUT_PNG,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    # --------------------------------------------------------
    # RINGKASAN AKHIR
    # --------------------------------------------------------

    before_counts = hitung_kelas(original)
    after_counts = hitung_kelas(hasil)

    print("\n" + "=" * 65)
    print("HASIL KLASIFIKASI FINAL")
    print("=" * 65)

    print(
        f"{'Kelas':<12}"
        f"{'Sebelum':>15}"
        f"{'Final':>15}"
        f"{'Perubahan':>15}"
    )

    print("-" * 57)

    for name in CLASS_CODES:
        before = before_counts[name]
        after = after_counts[name]

        print(
            f"{name:<12}"
            f"{before:>15,}"
            f"{after:>15,}"
            f"{after-before:>+15,}"
        )

    print("-" * 57)

    print("Raster:", OUTPUT_TIF)
    print("Visualisasi:", OUTPUT_PNG)

    print(
        "\nCatatan: hasil ini merupakan klasifikasi dengan "
        "filter spasial dan spektral. Validitas Mangrove "
        "tetap perlu diperiksa terhadap citra satelit."
    )

    print("=" * 65)
    print("SELESAI")
    print("=" * 65)


# ============================================================
# JALANKAN
# ============================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            "\nERROR:",
            type(error).__name__,
            "-",
            error,
        )
        raise