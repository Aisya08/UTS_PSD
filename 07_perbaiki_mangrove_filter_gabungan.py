import numpy as np
import rasterio

from scipy.ndimage import (
    label,
    distance_transform_edt,
    uniform_filter,
)


# ============================================================
# KONFIGURASI
# ============================================================

INPUT_FILE = "hasil_klasifikasi_jatim.tif"

OUTPUT_FILE = (
    "hasil_klasifikasi_jatim_mangrove_diperbaiki.tif"
)

BAND_B04 = (
    "2026-10-06-00_00_2026-10-06-23_59_"
    "Sentinel-2_L2A_B04_(Raw).tiff"
)

BAND_B08 = (
    "2026-10-06-00_00_2026-10-06-23_59_"
    "Sentinel-2_L2A_B08_(Raw).tiff"
)

# Kode kelas raster
BUILDING = 1
FARMLAND = 2
WATER = 3
FOREST = 4
MANGROVE = 5

# Parameter filter yang lebih longgar
MAX_DISTANCE_WATER_M = 2000
MAX_DISTANCE_LAND_M = 1000
MIN_NDVI = 0.15

# Jendela pencarian kelas tetangga
NEIGHBOR_WINDOW = 21

CLASS_NAMES = {
    1: "Building",
    2: "Farmland",
    3: "Water",
    4: "Forest",
    5: "Mangrove",
}


# ============================================================
# BACA RASTER
# ============================================================

def baca_raster(nama_file):
    with rasterio.open(nama_file) as src:
        data = src.read(1)
        profile = src.profile.copy()

    return data, profile


def hitung_pixel(data):
    return {
        nama: int(np.count_nonzero(data == kode))
        for kode, nama in CLASS_NAMES.items()
    }


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():

    print("=" * 65)
    print("FILTER GABUNGAN MANGROVE")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. BACA KLASIFIKASI ASLI
    # --------------------------------------------------------

    print("\n[1/7] Membaca klasifikasi asli...")

    with rasterio.open(INPUT_FILE) as src:
        original = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        bounds = src.bounds
        crs = src.crs

    if crs is None or crs.to_epsg() != 4326:
        raise ValueError(
            "Raster klasifikasi harus menggunakan EPSG:4326."
        )

    print("Ukuran raster:", original.shape)
    print("CRS:", crs)

    mangrove_awal = original == MANGROVE

    print(
        "Pixel Mangrove awal:",
        f"{np.count_nonzero(mangrove_awal):,}"
    )

    if not np.any(mangrove_awal):
        raise ValueError("Tidak ditemukan kelas Mangrove.")

    # --------------------------------------------------------
    # 2. BACA BAND B04 DAN B08
    # --------------------------------------------------------

    print("\n[2/7] Membaca band B04 dan B08...")

    with rasterio.open(BAND_B04) as src:
        b04 = src.read(1).astype(np.float32)

        if src.shape != original.shape:
            raise ValueError("Ukuran raster B04 tidak sesuai.")

        if src.crs != crs or src.transform != transform:
            raise ValueError(
                "CRS atau transform B04 tidak cocok."
            )

    with rasterio.open(BAND_B08) as src:
        b08 = src.read(1).astype(np.float32)

        if src.shape != original.shape:
            raise ValueError("Ukuran raster B08 tidak sesuai.")

        if src.crs != crs or src.transform != transform:
            raise ValueError(
                "CRS atau transform B08 tidak cocok."
            )

    # --------------------------------------------------------
    # 3. HITUNG NDVI
    # --------------------------------------------------------

    print("\n[3/7] Menghitung NDVI...")

    penyebut = b08 + b04

    ndvi = np.full(
        original.shape,
        np.nan,
        dtype=np.float32,
    )

    valid = (
        np.isfinite(b04)
        & np.isfinite(b08)
        & (penyebut > 0)
    )

    np.divide(
        b08 - b04,
        penyebut,
        out=ndvi,
        where=valid,
    )

    print("Ambang NDVI:", MIN_NDVI)

    # --------------------------------------------------------
    # 4. IDENTIFIKASI AIR TERBESAR
    # --------------------------------------------------------

    print("\n[4/7] Mengidentifikasi air terbesar...")

    water_mask = original == WATER

    if not np.any(water_mask):
        raise ValueError("Tidak ditemukan pixel Water.")

    struktur = np.ones((3, 3), dtype=np.uint8)

    label_air, jumlah_komponen = label(
        water_mask,
        structure=struktur,
    )

    ukuran = np.bincount(label_air.ravel())
    ukuran[0] = 0

    label_terbesar = int(np.argmax(ukuran))

    air_terbesar = label_air == label_terbesar

    print("Komponen air:", jumlah_komponen)
    print(
        "Pixel komponen terbesar:",
        f"{np.count_nonzero(air_terbesar):,}"
    )

    # --------------------------------------------------------
    # 5. HITUNG JARAK KE AIR DAN DARATAN
    # --------------------------------------------------------

    print("\n[5/7] Menghitung jarak spasial...")

    latitude_tengah = (
        bounds.top + bounds.bottom
    ) / 2

    pixel_y_m = abs(transform.e) * 111320

    pixel_x_m = (
        abs(transform.a)
        * 111320
        * np.cos(np.deg2rad(latitude_tengah))
    )

    if pixel_x_m <= 0 or pixel_y_m <= 0:
        raise ValueError("Resolusi raster tidak valid.")

    # Jarak ke komponen air terbesar.
    jarak_air = distance_transform_edt(
        ~air_terbesar,
        sampling=(pixel_y_m, pixel_x_m),
    )

    # Perkiraan daratan dari Building, Farmland, dan Forest.
    land_mask = np.isin(
        original,
        [BUILDING, FARMLAND, FOREST],
    )

    if not np.any(land_mask):
        raise ValueError("Tidak ditemukan kelas daratan.")

    jarak_darat = distance_transform_edt(
        ~land_mask,
        sampling=(pixel_y_m, pixel_x_m),
    )

    dekat_air = (
        jarak_air <= MAX_DISTANCE_WATER_M
    )

    dekat_darat = (
        jarak_darat <= MAX_DISTANCE_LAND_M
    )

    vegetasi = (
        np.isfinite(ndvi)
        & (ndvi >= MIN_NDVI)
    )

    # Tiga kriteria harus dipenuhi bersamaan.
    zona_valid = (
        dekat_air
        & dekat_darat
        & vegetasi
    )

    # Kandidat Mangrove yang tidak memenuhi kriteria.
    perlu_diperbaiki = (
        mangrove_awal
        & (~zona_valid)
    )

    print(
        "Batas jarak air:",
        f"{MAX_DISTANCE_WATER_M} meter"
    )

    print(
        "Batas jarak daratan:",
        f"{MAX_DISTANCE_LAND_M} meter"
    )

    print(
        "Mangrove yang dipertahankan:",
        f"{np.count_nonzero(mangrove_awal & zona_valid):,}"
    )

    print(
        "Mangrove yang akan diperbaiki:",
        f"{np.count_nonzero(perlu_diperbaiki):,}"
    )

    # --------------------------------------------------------
    # 6. GANTI DENGAN KELAS DOMINAN DI SEKITAR
    # --------------------------------------------------------

    print("\n[6/7] Menetapkan kelas pengganti...")

    hasil = original.copy()

    kelas_alternatif = [
        BUILDING,
        FARMLAND,
        WATER,
        FOREST,
    ]

    jumlah_tetangga = np.stack([
        uniform_filter(
            (original == kode).astype(np.float32),
            size=NEIGHBOR_WINDOW,
            mode="nearest",
        )
        for kode in kelas_alternatif
    ])

    # Kelas yang jumlahnya paling dominan di sekitar pixel.
    indeks_terbaik = np.argmax(
        jumlah_tetangga,
        axis=0,
    )

    kode_pengganti = np.asarray(
        kelas_alternatif,
        dtype=np.uint8,
    )[indeks_terbaik]

    hasil[perlu_diperbaiki] = (
        kode_pengganti[perlu_diperbaiki]
    )

    # Untuk kasus tidak ada kelas pengganti di sekitarnya,
    # gunakan kelas non-Mangrove terdekat.
    tidak_ada_tetangga = (
        perlu_diperbaiki
        & (np.max(jumlah_tetangga, axis=0) == 0)
    )

    if np.any(tidak_ada_tetangga):

        kelas_valid = np.isin(
            original,
            kelas_alternatif,
        )

        indeks_terdekat = distance_transform_edt(
            ~kelas_valid,
            return_distances=False,
            return_indices=True,
        )

        kelas_terdekat = original[
            tuple(indeks_terdekat)
        ]

        hasil[tidak_ada_tetangga] = (
            kelas_terdekat[tidak_ada_tetangga]
        )

    # --------------------------------------------------------
    # 7. SIMPAN HASIL DAN CETAK RINGKASAN
    # --------------------------------------------------------

    print("\n[7/7] Menyimpan raster...")

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

    sebelum = hitung_pixel(original)
    sesudah = hitung_pixel(hasil)

    print("\n" + "=" * 65)
    print("PERBANDINGAN HASIL")
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
    print("File hasil:", OUTPUT_FILE)
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
            error,
        )
        raise