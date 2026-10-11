import numpy as np
import pandas as pd
import rasterio
import joblib
import matplotlib.pyplot as plt

from glob import glob
from scipy.ndimage import label, distance_transform_edt
from matplotlib.colors import ListedColormap, BoundaryNorm


# ============================================================
# 1. KONFIGURASI
# ============================================================

CLASS_FILE = "hasil_klasifikasi_jatim.tif"
MODEL_FILE = "random_forest_jatim.pkl"

OUTPUT_FILE = (
    "hasil_klasifikasi_jatim_mangrove_koridor_pesisir.tif"
)

OUTPUT_IMAGE = "evaluasi_mangrove_koridor_pesisir.png"

FEATURES = ["B04", "B08", "B8A", "B11"]

BUILDING = 1
FARMLAND = 2
WATER = 3
FOREST = 4
MANGROVE = 5

CLASS_CODES = {
    "Building": BUILDING,
    "Farmland": FARMLAND,
    "Water": WATER,
    "Forest": FOREST,
    "Mangrove": MANGROVE,
}

# Parameter percobaan
MNDWI_THRESHOLD = 0.05
WATER_B08_PERCENTILE = 90

COASTAL_BUFFER_M = 1000
MANGROVE_MNDWI_MAX = 0.10

PREDICTION_BATCH_SIZE = 50000


# ============================================================
# 2. FUNGSI BANTU
# ============================================================

def cari_band(nama_band):
    pola = [
        f"*Sentinel-2_L2A_{nama_band}_(Raw).tiff",
        f"*Sentinel-2_L2A_{nama_band}_(Raw).tif",
    ]

    for pattern in pola:
        files = sorted(glob(pattern))

        if files:
            return files[0]

    raise FileNotFoundError(
        f"Band {nama_band} tidak ditemukan."
    )


def baca_band(
    nama_band,
    shape_target,
    crs_target,
    transform_target
):
    path = cari_band(nama_band)

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)

        if src.shape != shape_target:
            raise ValueError(
                f"Ukuran band {nama_band} tidak cocok."
            )

        if src.crs != crs_target:
            raise ValueError(
                f"CRS band {nama_band} tidak cocok."
            )

        if src.transform != transform_target:
            raise ValueError(
                f"Transform band {nama_band} tidak cocok."
            )

    print(f"{nama_band}: {path}")

    return data


def cari_komponen_tepi(mask):
    """Mengidentifikasi komponen yang menyentuh tepi citra."""

    struktur = np.ones((3, 3), dtype=np.uint8)

    labels, jumlah_komponen = label(
        mask.astype(bool),
        structure=struktur,
    )

    label_tepi = np.unique(
        np.concatenate([
            labels[0, :],
            labels[-1, :],
            labels[:, 0],
            labels[:, -1],
        ])
    )

    label_tepi = label_tepi[label_tepi > 0]

    if label_tepi.size == 0:
        return None, jumlah_komponen

    mask_tepi = np.isin(labels, label_tepi)

    return mask_tepi, jumlah_komponen


def normalisasi(data):
    """Normalisasi untuk visualisasi saja."""

    valid = np.isfinite(data)

    if not np.any(valid):
        return np.zeros_like(data, dtype=np.float32)

    values = data[valid]
    bawah, atas = np.percentile(values, [2, 98])

    if atas <= bawah:
        return np.zeros_like(data, dtype=np.float32)

    hasil = (data - bawah) / (atas - bawah)

    return np.clip(hasil, 0, 1)


def hitung_jumlah_kelas(data):
    return {
        nama: int(np.count_nonzero(data == kode))
        for nama, kode in CLASS_CODES.items()
    }


# ============================================================
# 3. PROGRAM UTAMA
# ============================================================

def main():

    print("=" * 65)
    print("PERBAIKAN MANGROVE DI PESISIR DAN PERAIRAN")
    print("=" * 65)

    # --------------------------------------------------------
    # LANGKAH 1: BACA KLASIFIKASI ASLI
    # --------------------------------------------------------

    print("\n[1/7] Membaca klasifikasi asli...")

    with rasterio.open(CLASS_FILE) as src:
        original = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        bounds = src.bounds
        crs = src.crs

    if crs is None or crs.to_epsg() != 4326:
        raise ValueError(
            "Raster harus menggunakan EPSG:4326."
        )

    shape_target = original.shape

    print("Ukuran:", shape_target)
    print("CRS:", crs)

    print(
        "Mangrove awal:",
        f"{np.count_nonzero(original == MANGROVE):,}"
    )

    # --------------------------------------------------------
    # LANGKAH 2: BACA SEMUA BAND
    # --------------------------------------------------------

    print("\n[2/7] Membaca band Sentinel-2...")

    bands = {}

    for nama_band in ["B03", "B04", "B08", "B8A", "B11"]:
        bands[nama_band] = baca_band(
            nama_band,
            shape_target,
            crs,
            transform,
        )

    b03 = bands["B03"]
    b04 = bands["B04"]
    b08 = bands["B08"]
    b8a = bands["B8A"]
    b11 = bands["B11"]

    # --------------------------------------------------------
    # LANGKAH 3: HITUNG MNDWI
    # --------------------------------------------------------

    print("\n[3/7] Menghitung MNDWI...")

    penyebut = b03 + b11

    valid = (
        np.isfinite(b03)
        & np.isfinite(b04)
        & np.isfinite(b08)
        & np.isfinite(b8a)
        & np.isfinite(b11)
        & (b03 > 0)
        & (b08 > 0)
        & (b11 > 0)
        & (penyebut > 0)
    )

    mndwi = np.full(
        shape_target,
        np.nan,
        dtype=np.float32,
    )

    np.divide(
        b03 - b11,
        penyebut,
        out=mndwi,
        where=valid,
    )

    # Persentil B08 piksel yang sudah diklasifikasikan Water.
    water_asli = (
        (original == WATER)
        & valid
    )

    if not np.any(water_asli):
        raise RuntimeError(
            "Tidak ditemukan piksel Water sebagai acuan."
        )

    ambang_b08 = float(
        np.percentile(
            b08[water_asli],
            WATER_B08_PERCENTILE,
        )
    )

    print("Ambang MNDWI:", MNDWI_THRESHOLD)
    print("Ambang B08:", round(ambang_b08, 5))

    # --------------------------------------------------------
    # LANGKAH 4: DETEKSI LAUT
    # --------------------------------------------------------

    print("\n[4/7] Mendeteksi area laut...")

    kandidat_air = (
        valid
        & (mndwi > MNDWI_THRESHOLD)
        & (b08 <= ambang_b08)
    )

    laut, jumlah_komponen = cari_komponen_tepi(
        kandidat_air
    )

    metode_laut = "MNDWI + B08"

    if laut is None or not np.any(laut):

        print(
            "Deteksi MNDWI gagal. "
            "Mencoba kelas Water asli yang menyentuh tepi."
        )

        laut, jumlah_komponen = cari_komponen_tepi(
            original == WATER
        )

        metode_laut = "Water asli di tepi citra"

    if laut is None or not np.any(laut):

        print(
            "Komponen Water di tepi tidak ditemukan. "
            "Menggunakan komponen Water terbesar."
        )

        labels, jumlah_komponen = label(
            original == WATER,
            structure=np.ones((3, 3), dtype=np.uint8),
        )

        ukuran = np.bincount(labels.ravel())
        ukuran[0] = 0

        if ukuran.size <= 1 or np.max(ukuran) == 0:
            raise RuntimeError(
                "Tidak ditemukan komponen Water."
            )

        laut = labels == int(np.argmax(ukuran))

        metode_laut = "Komponen Water terbesar"

    jumlah_pixel_laut = int(np.count_nonzero(laut))

    print("Metode deteksi:", metode_laut)
    print(
        "Pixel laut terdeteksi:",
        f"{jumlah_pixel_laut:,}"
    )

    # --------------------------------------------------------
    # LANGKAH 5: KORIDOR PESISIR DAN PIXEL AIR
    # --------------------------------------------------------

    print("\n[5/7] Menentukan Mangrove pesisir...")

    latitude_tengah = (
        bounds.top + bounds.bottom
    ) / 2.0

    pixel_y_m = abs(transform.e) * 111320.0

    pixel_x_m = (
        abs(transform.a)
        * 111320.0
        * np.cos(np.deg2rad(latitude_tengah))
    )

    if pixel_x_m <= 0 or pixel_y_m <= 0:
        raise ValueError(
            "Resolusi raster tidak valid."
        )

    jarak_laut = distance_transform_edt(
        ~laut,
        sampling=(pixel_y_m, pixel_x_m),
    )

    koridor_pesisir = (
        jarak_laut <= COASTAL_BUFFER_M
    )

    mangrove_asli = original == MANGROVE

    # Pixel Mangrove yang spektrumnya cenderung seperti air.
    mangrove_mirip_air = (
        mangrove_asli
        & valid
        & (mndwi > 0.0)
        & (b08 <= ambang_b08)
    )

    # Mangrove dipertahankan jika berada di koridor,
    # tidak berada pada mask laut, serta tidak terlalu
    # menyerupai air terbuka.
    pertahankan_mangrove = (
        mangrove_asli
        & koridor_pesisir
        & (~laut)
        & valid
        & (mndwi <= MANGROVE_MNDWI_MAX)
        & (~mangrove_mirip_air)
    )

    mangrove_dikeluarkan = (
        mangrove_asli
        & (~pertahankan_mangrove)
    )

    print(
        "Batas koridor:",
        COASTAL_BUFFER_M,
        "meter"
    )

    print(
        "Mangrove dipertahankan:",
        f"{np.count_nonzero(pertahankan_mangrove):,}"
    )

    print(
        "Mangrove yang direklasifikasi:",
        f"{np.count_nonzero(mangrove_dikeluarkan):,}"
    )

    print(
        "Mangrove dengan karakteristik air:",
        f"{np.count_nonzero(mangrove_mirip_air):,}"
    )

    # --------------------------------------------------------
    # LANGKAH 6: REKLASIFIKASI MANGROVE YANG DIKELUARKAN
    # --------------------------------------------------------

    print("\n[6/7] Mereklasifikasi piksel...")

    hasil = original.copy()

    model = joblib.load(MODEL_FILE)

    kelas_model = [str(c) for c in model.classes_]

    indeks_non_mangrove = [
        i
        for i, nama in enumerate(kelas_model)
        if nama != "Mangrove"
    ]

    if not indeks_non_mangrove:
        raise RuntimeError(
            "Model tidak memiliki kelas alternatif."
        )

    kode_kelas_model = np.asarray([
        CLASS_CODES[nama]
        for nama in kelas_model
    ], dtype=np.uint8)

    indeks_pixel = np.flatnonzero(
        mangrove_dikeluarkan.ravel()
    )

    jumlah_batch = (
        len(indeks_pixel) + PREDICTION_BATCH_SIZE - 1
    ) // PREDICTION_BATCH_SIZE

    for nomor, awal in enumerate(
        range(
            0,
            len(indeks_pixel),
            PREDICTION_BATCH_SIZE,
        ),
        start=1,
    ):

        indeks_batch = indeks_pixel[
            awal:awal + PREDICTION_BATCH_SIZE
        ]

        X = pd.DataFrame({
            nama: bands[nama].ravel()[indeks_batch]
            for nama in FEATURES
        })

        probabilities = model.predict_proba(X)

        probabilities_non_mangrove = probabilities[
            :, indeks_non_mangrove
        ]

        posisi_terbaik = np.argmax(
            probabilities_non_mangrove,
            axis=1,
        )

        posisi_model = np.asarray(
            indeks_non_mangrove
        )[posisi_terbaik]

        kode_pengganti = kode_kelas_model[
            posisi_model
        ]

        hasil.ravel()[indeks_batch] = kode_pengganti

        print(
            f"Batch {nomor}/{jumlah_batch} selesai."
        )

    # --------------------------------------------------------
    # ATURAN PENTING:
    # PIKSEL LAUT YANG TERDETEKSI DIPAKSA MENJADI WATER.
    # --------------------------------------------------------

    print("\nMenerapkan mask laut sebagai kelas Water...")

    hasil[laut] = WATER

    # Pixel Mangrove yang jelas menyerupai air juga dialihkan.
    hasil[mangrove_mirip_air] = WATER

    # Kelas Mangrove pesisir yang lolos semua kriteria.
    # Area laut tidak boleh dipertahankan sebagai Mangrove.
    hasil[pertahankan_mangrove & (~laut)] = MANGROVE

    # --------------------------------------------------------
    # SIMPAN HASIL BARU
    # --------------------------------------------------------

    profile.update(
        dtype="uint8",
        count=1,
        compress="lzw",
    )

    with rasterio.open(
        OUTPUT_FILE,
        "w",
        **profile,
    ) as dst:
        dst.write(hasil.astype(np.uint8), 1)

    print("Raster baru:", OUTPUT_FILE)

    # --------------------------------------------------------
    # VISUALISASI
    # --------------------------------------------------------

    print("\nMembuat gambar perbandingan...")

    false_color = np.dstack([
        normalisasi(b11),
        normalisasi(b08),
        normalisasi(b04),
    ])

    cmap_kelas = ListedColormap([
        "#FFFFFF",
        "#E53935",
        "#FDD835",
        "#1E88E5",
        "#43A047",
        "#8E24AA",
    ])

    norm_kelas = BoundaryNorm(
        np.arange(-0.5, 6.5, 1),
        cmap_kelas.N,
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(22, 7),
    )

    axes[0].imshow(false_color)
    axes[0].set_title("Sentinel-2 False Color")
    axes[0].axis("off")

    axes[1].imshow(
        laut,
        cmap=ListedColormap(["#EEEEEE", "#168BFF"]),
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Mask Laut")
    axes[1].axis("off")

    axes[2].imshow(
        original,
        cmap=cmap_kelas,
        norm=norm_kelas,
        interpolation="nearest",
    )
    axes[2].set_title("Klasifikasi Asli")
    axes[2].axis("off")

    axes[3].imshow(
        hasil,
        cmap=cmap_kelas,
        norm=norm_kelas,
        interpolation="nearest",
    )
    axes[3].set_title("Mangrove Setelah Perbaikan")
    axes[3].axis("off")

    fig.suptitle(
        "Evaluasi Mangrove dan Mask Laut",
        fontsize=15,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_IMAGE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.show()

    # --------------------------------------------------------
    # LAPORAN AKHIR
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("PERBANDINGAN KELAS")
    print("=" * 65)

    sebelum = hitung_jumlah_kelas(original)
    sesudah = hitung_jumlah_kelas(hasil)

    print(
        f"{'Kelas':<12}"
        f"{'Sebelum':>15}"
        f"{'Sesudah':>15}"
        f"{'Perubahan':>15}"
    )

    print("-" * 57)

    for nama in CLASS_CODES:
        awal = sebelum[nama]
        akhir = sesudah[nama]

        print(
            f"{nama:<12}"
            f"{awal:>15,}"
            f"{akhir:>15,}"
            f"{akhir - awal:>+15,}"
        )

    print("-" * 57)

    print("File raster:", OUTPUT_FILE)
    print("Gambar evaluasi:", OUTPUT_IMAGE)

    print(
        "\nCatatan: mask laut merupakan estimasi spektral. "
        "Periksa panel Mask Laut; piksel daratan yang "
        "salah masuk mask juga dapat berubah menjadi Water."
    )

    print("=" * 65)
    print("SELESAI")
    print("=" * 65)


# ============================================================
# JALANKAN PROGRAM
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except FileNotFoundError as error:
        print("\nERROR: File tidak ditemukan.")
        print(error)

    except Exception as error:
        print(
            "\nERROR:",
            type(error).__name__,
            "-",
            error
        )
        raise