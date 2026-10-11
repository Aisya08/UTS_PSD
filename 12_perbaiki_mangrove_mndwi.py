import numpy as np
import rasterio
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm


# ============================================================
# KONFIGURASI
# ============================================================

INPUT_FILE = "hasil_klasifikasi_jatim.tif"

OUTPUT_FILE = "hasil_klasifikasi_jatim_mangrove_mndwi.tif"

OUTPUT_IMAGE = "perbandingan_mangrove_mndwi.png"

B04_FILE = (
    "2026-10-06-00_00_2026-10-06-23_59_"
    "Sentinel-2_L2A_B04_(Raw).tiff"
)

B03_FILE = (
    "2026-10-06-00_00_2026-10-06-23_59_"
    "Sentinel-2_L2A_B03_(Raw).tiff"
)

B08_FILE = (
    "2026-10-06-00_00_2026-10-06-23_59_"
    "Sentinel-2_L2A_B08_(Raw).tiff"
)

B11_FILE = (
    "2026-10-06-00_00_2026-10-06-23_59_"
    "Sentinel-2_L2A_B11_(Raw).tiff"
)

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
    5: "Mangrove",
}

# Ambang awal MNDWI. Nilai ini perlu dievaluasi
# terhadap karakteristik citra yang digunakan.
MIN_MNDWI = 0.05

# Persentil B08 dari piksel yang saat ini berlabel Water.
# Dipakai untuk menekan kemungkinan awan ikut terdeteksi.
WATER_B08_PERCENTILE = 90


# ============================================================
# FUNGSI BANTU
# ============================================================

def baca_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    return data, profile


def normalisasi(data):
    valid = np.isfinite(data)

    if not np.any(valid):
        return np.zeros_like(data, dtype=np.float32)

    nilai = data[valid]
    bawah, atas = np.percentile(nilai, [2, 98])

    if atas <= bawah:
        return np.zeros_like(data, dtype=np.float32)

    hasil = (data - bawah) / (atas - bawah)

    return np.clip(hasil, 0, 1)


def hitung_jumlah_kelas(data):
    return {
        nama: int(np.count_nonzero(data == kode))
        for kode, nama in CLASS_NAMES.items()
    }


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():

    print("=" * 65)
    print("PERBAIKAN MANGROVE DENGAN MNDWI")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. BACA RASTER KLASIFIKASI ASLI
    # --------------------------------------------------------

    print("\n[1/6] Membaca klasifikasi asli...")

    with rasterio.open(INPUT_FILE) as src:
        original = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

    print("Ukuran:", original.shape)
    print("CRS:", crs)

    if crs is None:
        raise ValueError("CRS raster klasifikasi tidak tersedia.")

    # --------------------------------------------------------
    # 2. BACA BAND SENTINEL-2
    # --------------------------------------------------------

    print("\n[2/6] Membaca band B03, B04, B08, dan B11...")

    b03, p03 = baca_raster(B03_FILE)
    b04, p04 = baca_raster(B04_FILE)
    b08, p08 = baca_raster(B08_FILE)
    b11, p11 = baca_raster(B11_FILE)

    for nama, profil_band in [
        ("B03", p03),
        ("B04", p04),
        ("B08", p08),
        ("B11", p11),
    ]:
        if b03.shape != original.shape:
            raise ValueError(
                "Ukuran band tidak cocok dengan klasifikasi."
            )

        if (
            profil_band["crs"] != crs
            or profil_band["transform"] != transform
        ):
            raise ValueError(
                f"CRS atau transform {nama} tidak cocok "
                "dengan klasifikasi."
            )

    if not (
        b03.shape == b04.shape == b08.shape == b11.shape
    ):
        raise ValueError("Ukuran antarbands berbeda.")

    # --------------------------------------------------------
    # 3. HITUNG MNDWI
    # --------------------------------------------------------

    print("\n[3/6] Menghitung MNDWI...")

    penyebut = b03 + b11

    valid = (
        np.isfinite(b03)
        & np.isfinite(b08)
        & np.isfinite(b11)
        & (b03 >= 0)
        & (b08 >= 0)
        & (b11 >= 0)
        & (b03 + b11 > 0)
    )

    mndwi = np.full(
        original.shape,
        np.nan,
        dtype=np.float32,
    )

    np.divide(
        b03 - b11,
        penyebut,
        out=mndwi,
        where=valid,
    )

    print("Median B03:", float(np.median(b03)))
    print("Median B08:", float(np.median(b08)))
    print("Median B11:", float(np.median(b11)))

    # --------------------------------------------------------
    # 4. TENTUKAN KRITERIA PIXEL YANG MENYERUPAI AIR
    # --------------------------------------------------------

    print("\n[4/6] Memilih kandidat Mangrove yang menyerupai air...")

    mangrove_awal = original == MANGROVE
    water_awal = (original == WATER) & valid

    jumlah_mangrove_awal = int(
        np.count_nonzero(mangrove_awal)
    )

    if not np.any(water_awal):
        raise ValueError(
            "Tidak ditemukan piksel Water untuk pembanding."
        )

    # Gunakan distribusi nilai B08 pada piksel Water
    # sebagai acuan awal untuk menekan piksel awan.
    nilai_b08_water = b08[water_awal]

    ambang_b08 = float(
        np.percentile(
            nilai_b08_water,
            WATER_B08_PERCENTILE,
        )
    )

    print("Ambang minimum MNDWI:", MIN_MNDWI)
    print(
        f"Ambang B08 P{WATER_B08_PERCENTILE} Water:",
        round(ambang_b08, 4),
    )

    # Hanya pixel Mangrove yang memiliki karakteristik
    # mirip air pada kedua ukuran ini yang diubah.
    kandidat_air = (
        mangrove_awal
        & valid
        & (mndwi > MIN_MNDWI)
        & (b08 <= ambang_b08)
    )

    jumlah_kandidat = int(
        np.count_nonzero(kandidat_air)
    )

    print(
        "Pixel Mangrove yang memenuhi kriteria:",
        f"{jumlah_kandidat:,}",
    )

    # --------------------------------------------------------
    # 5. SIMPAN HASIL KE FILE BARU
    # --------------------------------------------------------

    print("\n[5/6] Membuat klasifikasi hasil perbaikan...")

    hasil = original.copy()

    # Reclassify kandidat menjadi Water.
    hasil[kandidat_air] = WATER

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

    print("Raster hasil:", OUTPUT_FILE)

    # --------------------------------------------------------
    # 6. TAMPILKAN PERBANDINGAN
    # --------------------------------------------------------

    print("\n[6/6] Membuat gambar perbandingan...")

    sebelum = hitung_jumlah_kelas(original)
    sesudah = hitung_jumlah_kelas(hasil)

    print("\n" + "=" * 65)
    print("PERBANDINGAN JUMLAH PIXEL")
    print("=" * 65)

    print(
        f"{'Kelas':<12}"
        f"{'Sebelum':>15}"
        f"{'Sesudah':>15}"
        f"{'Perubahan':>15}"
    )

    print("-" * 57)

    for kode, nama in CLASS_NAMES.items():
        awal = sebelum[nama]
        akhir = sesudah[nama]

        print(
            f"{nama:<12}"
            f"{awal:>15,}"
            f"{akhir:>15,}"
            f"{akhir-awal:>+15,}"
        )

    print("-" * 57)

    # False color RGB = B11, B08, B04
    false_color = np.dstack([
        normalisasi(b11),
        normalisasi(b08),
        normalisasi(b04),
    ])

    warna_kelas = ListedColormap([
        "#FFFFFF",
        "#E53935",
        "#FDD835",
        "#1E88E5",
        "#43A047",
        "#8E24AA",
    ])

    norm_kelas = BoundaryNorm(
        np.arange(-0.5, 6.5, 1),
        warna_kelas.N,
    )

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 7),
    )

    axes[0].imshow(false_color)
    axes[0].set_title("Sentinel-2 False Color")
    axes[0].axis("off")

    axes[1].imshow(
        original,
        cmap=warna_kelas,
        norm=norm_kelas,
        interpolation="nearest",
    )
    axes[1].set_title("Klasifikasi Asli")
    axes[1].axis("off")

    axes[2].imshow(
        hasil,
        cmap=warna_kelas,
        norm=norm_kelas,
        interpolation="nearest",
    )
    axes[2].set_title("Setelah Filter MNDWI + B08")
    axes[2].axis("off")

    fig.suptitle(
        "Perbandingan Perbaikan Klasifikasi Mangrove",
        fontsize=15,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_IMAGE,
        dpi=180,
        bbox_inches="tight",
    )

    print("\nGambar perbandingan:", OUTPUT_IMAGE)

    plt.show()

    print("\nSelesai.")
    print(
        "Klasifikasi asli tidak ditimpa. "
        "Hasil baru perlu diperiksa sebelum digunakan."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nERROR:", type(error).__name__, "-", error)
        raise