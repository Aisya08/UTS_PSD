import glob
import numpy as np
import rasterio


# ============================================================
# KONFIGURASI
# ============================================================

INPUT_FILE = "hasil_klasifikasi_jatim.tif"

OUTPUT_FILE = (
    "hasil_klasifikasi_jatim_mangrove_filter_ndvi.tif"
)

# Kode kelas
WATER = 3
MANGROVE = 5

# Ambang awal NDVI untuk mengidentifikasi piksel
# Mangrove yang mungkin sebenarnya merupakan air.
NDVI_THRESHOLD = 0.15

# Persentil ke-90 nilai B08 pada piksel yang sudah
# diklasifikasikan sebagai Water.
# Dipakai sebagai pembatas tambahan.
WATER_B08_PERCENTILE = 90


# ============================================================
# CARI BAND SENTINEL-2
# ============================================================

def cari_band(nama_band):
    pola = [
        f"*Sentinel-2_L2A_{nama_band}_(Raw).tiff",
        f"*Sentinel-2_L2A_{nama_band}_(Raw).tif",
    ]

    for pattern in pola:
        hasil = glob.glob(pattern)

        if hasil:
            return hasil[0]

    raise FileNotFoundError(
        f"File band {nama_band} tidak ditemukan."
    )


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():

    print("=" * 65)
    print("FILTER MANGROVE BERDASARKAN NILAI SPEKTRAL")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. BACA KLASIFIKASI ASLI
    # --------------------------------------------------------

    print("\n[1/5] Membaca klasifikasi asli...")

    with rasterio.open(INPUT_FILE) as src:
        klasifikasi = src.read(1)
        profile = src.profile.copy()
        transform_asli = src.transform
        crs_asli = src.crs

    print("Input:", INPUT_FILE)
    print("Ukuran:", klasifikasi.shape)
    print("CRS:", crs_asli)

    mangrove_awal = klasifikasi == MANGROVE
    water_awal = klasifikasi == WATER

    print(
        "Pixel Mangrove awal:",
        f"{np.count_nonzero(mangrove_awal):,}"
    )

    print(
        "Pixel Water awal:",
        f"{np.count_nonzero(water_awal):,}"
    )

    if not np.any(mangrove_awal):
        raise ValueError(
            "Tidak ditemukan kelas Mangrove."
        )

    if not np.any(water_awal):
        raise ValueError(
            "Tidak ditemukan kelas Water untuk pembanding."
        )

    # --------------------------------------------------------
    # 2. BACA BAND B04 DAN B08
    # --------------------------------------------------------

    print("\n[2/5] Membaca band B04 dan B08...")

    file_b04 = cari_band("B04")
    file_b08 = cari_band("B08")

    with rasterio.open(file_b04) as src:
        b04 = src.read(1).astype(np.float32)

        if src.shape != klasifikasi.shape:
            raise ValueError(
                "Ukuran B04 berbeda dari klasifikasi."
            )

        if src.crs != crs_asli:
            raise ValueError(
                "CRS B04 berbeda dari klasifikasi."
            )

        if src.transform != transform_asli:
            raise ValueError(
                "Transform B04 berbeda dari klasifikasi."
            )

    with rasterio.open(file_b08) as src:
        b08 = src.read(1).astype(np.float32)

        if src.shape != klasifikasi.shape:
            raise ValueError(
                "Ukuran B08 berbeda dari klasifikasi."
            )

        if src.crs != crs_asli:
            raise ValueError(
                "CRS B08 berbeda dari klasifikasi."
            )

        if src.transform != transform_asli:
            raise ValueError(
                "Transform B08 berbeda dari klasifikasi."
            )

    print("B04:", file_b04)
    print("B08:", file_b08)

    # --------------------------------------------------------
    # 3. HITUNG NDVI
    # --------------------------------------------------------

    print("\n[3/5] Menghitung NDVI...")

    penyebut = b08 + b04

    ndvi = np.full(
        klasifikasi.shape,
        np.nan,
        dtype=np.float32
    )

    valid = penyebut > 0

    np.divide(
        b08 - b04,
        penyebut,
        out=ndvi,
        where=valid
    )

    # NDVI Mangrove sebelum perbaikan
    ndvi_mangrove = ndvi[
        mangrove_awal & np.isfinite(ndvi)
    ]

    ndvi_water = ndvi[
        water_awal & np.isfinite(ndvi)
    ]

    print("\nStatistik NDVI kelas Mangrove:")
    if ndvi_mangrove.size:
        print(
            "Minimum :",
            round(float(np.min(ndvi_mangrove)), 3)
        )
        print(
            "Median  :",
            round(float(np.median(ndvi_mangrove)), 3)
        )
        print(
            "Maksimum:",
            round(float(np.max(ndvi_mangrove)), 3)
        )

    print("\nStatistik NDVI kelas Water:")
    if ndvi_water.size:
        print(
            "Median  :",
            round(float(np.median(ndvi_water)), 3)
        )
        print(
            "Persentil 90:",
            round(
                float(np.percentile(ndvi_water, 90)),
                3
            )
        )

    # --------------------------------------------------------
    # 4. IDENTIFIKASI MANGROVE YANG MENYERUPAI AIR
    # --------------------------------------------------------

    print("\n[4/5] Mengidentifikasi kandidat salah klasifikasi...")

    # Nilai B08 maksimum berdasarkan persentil kelas Water.
    nilai_b08_water = b08[
        water_awal & np.isfinite(b08)
    ]

    ambang_b08_water = float(
        np.percentile(
            nilai_b08_water,
            WATER_B08_PERCENTILE
        )
    )

    print(
        f"Ambang NDVI: {NDVI_THRESHOLD:.2f}"
    )

    print(
        f"Ambang B08 (P{WATER_B08_PERCENTILE} kelas Water):",
        round(ambang_b08_water, 3)
    )

    # Piksel harus memenuhi kedua kondisi:
    # 1. Diprediksi sebagai Mangrove.
    # 2. NDVI rendah dan nilai B08 berada dalam
    #    rentang yang relatif rendah seperti kelas Water.
    kandidat_air = (
        mangrove_awal
        & np.isfinite(ndvi)
        & (ndvi < NDVI_THRESHOLD)
        & (b08 <= ambang_b08_water)
    )

    jumlah_kandidat = int(
        np.count_nonzero(kandidat_air)
    )

    print(
        "Kandidat Mangrove yang menyerupai air:",
        f"{jumlah_kandidat:,}",
        "pixel"
    )

    # --------------------------------------------------------
    # 5. SIMPAN HASIL BARU
    # --------------------------------------------------------

    print("\n[5/5] Menyimpan raster baru...")

    hasil = klasifikasi.copy()

    # Hanya kandidat yang memenuhi kedua kriteria
    # yang diubah menjadi Water.
    hasil[kandidat_air] = WATER

    with rasterio.open(
        OUTPUT_FILE,
        "w",
        **profile
    ) as dst:
        dst.write(hasil, 1)

    print("\nPerbandingan jumlah pixel:")

    for nama, kode in [
        ("Building", 1),
        ("Farmland", 2),
        ("Water", 3),
        ("Forest", 4),
        ("Mangrove", 5),
    ]:
        sebelum = int(
            np.count_nonzero(klasifikasi == kode)
        )

        sesudah = int(
            np.count_nonzero(hasil == kode)
        )

        print(
            f"{nama:10s}: "
            f"{sebelum:>9,} -> {sesudah:>9,}"
        )

    print("\nFile hasil:", OUTPUT_FILE)

    print(
        "\nCatatan: ambang spektral ini merupakan "
        "percobaan awal, bukan bukti pasti bahwa "
        "setiap piksel yang diubah adalah air."
    )

    print("=" * 65)
    print("SELESAI")
    print("=" * 65)


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            "\nERROR:",
            type(error).__name__,
            "-",
            error
        )
        raise