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

CLASS_CODES = {
    "Building": 1,
    "Farmland": 2,
    "Water": 3,
    "Forest": 4,
    "Mangrove": 5,
}

BUILDING = CLASS_CODES["Building"]
FARMLAND = CLASS_CODES["Farmland"]
WATER = CLASS_CODES["Water"]
FOREST = CLASS_CODES["Forest"]
MANGROVE = CLASS_CODES["Mangrove"]

# Mangrove harus dekat air dan daratan.
MAX_DISTANCE_WATER_M = 500
MAX_DISTANCE_LAND_M = 200

# Jendela untuk mencari kelas dominan di sekitar pixel.
NEIGHBOR_WINDOW = 21


# ============================================================
# FUNGSI
# ============================================================

def hitung_pixel(data):
    """Menghitung jumlah pixel setiap kelas."""
    return {
        nama: int(np.count_nonzero(data == kode))
        for nama, kode in CLASS_CODES.items()
    }


def main():

    print("=" * 65)
    print("PERBAIKAN MANGROVE: FILTER AIR DAN DARATAN")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. BACA KLASIFIKASI ASLI
    # --------------------------------------------------------

    print("\n[1/6] Membaca klasifikasi asli...")

    with rasterio.open(INPUT_FILE) as src:
        original = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        bounds = src.bounds
        crs = src.crs

    if crs is None or crs.to_epsg() != 4326:
        raise ValueError(
            "Raster harus menggunakan CRS EPSG:4326."
        )

    print("File:", INPUT_FILE)
    print("Ukuran raster:", original.shape)
    print("CRS:", crs)

    mangrove_awal = int(
        np.count_nonzero(original == MANGROVE)
    )

    print("Pixel Mangrove awal:", f"{mangrove_awal:,}")

    if mangrove_awal == 0:
        raise RuntimeError(
            "Tidak ditemukan pixel Mangrove."
        )

    # --------------------------------------------------------
    # 2. CARI KOMPONEN AIR TERBESAR
    # --------------------------------------------------------

    print("\n[2/6] Mencari area air terbesar...")

    water_mask = original == WATER

    if not np.any(water_mask):
        raise RuntimeError(
            "Tidak ditemukan pixel Water."
        )

    struktur = np.ones((3, 3), dtype=np.uint8)

    label_air, jumlah_komponen = label(
        water_mask,
        structure=struktur
    )

    ukuran = np.bincount(label_air.ravel())
    ukuran[0] = 0

    label_air_terbesar = int(np.argmax(ukuran))

    air_terbesar = (
        label_air == label_air_terbesar
    )

    print("Komponen air:", jumlah_komponen)
    print(
        "Pixel area air terbesar:",
        f"{np.count_nonzero(air_terbesar):,}"
    )

    # --------------------------------------------------------
    # 3. HITUNG JARAK KE AIR DAN DARATAN
    # --------------------------------------------------------

    print("\n[3/6] Menghitung jarak ke air dan daratan...")

    lat_tengah = (bounds.top + bounds.bottom) / 2

    pixel_y_m = abs(transform.e) * 111320

    pixel_x_m = (
        abs(transform.a)
        * 111320
        * np.cos(np.deg2rad(lat_tengah))
    )

    if pixel_x_m <= 0 or pixel_y_m <= 0:
        raise ValueError(
            "Resolusi raster tidak valid."
        )

    print(f"Resolusi X: {pixel_x_m:.2f} meter/pixel")
    print(f"Resolusi Y: {pixel_y_m:.2f} meter/pixel")

    # Jarak ke komponen air terbesar.
    jarak_air = distance_transform_edt(
        ~air_terbesar,
        sampling=(pixel_y_m, pixel_x_m)
    )

    # Daratan diperkirakan dari kelas Building,
    # Farmland, dan Forest.
    #
    # Mangrove dan Water tidak dianggap sebagai
    # daratan dalam perhitungan ini.
    land_mask = np.isin(
        original,
        [BUILDING, FARMLAND, FOREST]
    )

    if not np.any(land_mask):
        raise RuntimeError(
            "Tidak ditemukan pixel kelas daratan."
        )

    jarak_darat = distance_transform_edt(
        ~land_mask,
        sampling=(pixel_y_m, pixel_x_m)
    )

    # Syarat pertama: dekat air.
    dekat_air = (
        jarak_air <= MAX_DISTANCE_WATER_M
    )

    # Syarat kedua: dekat daratan.
    dekat_darat = (
        jarak_darat <= MAX_DISTANCE_LAND_M
    )

    # Mangrove hanya dipertahankan jika memenuhi
    # KEDUA syarat secara bersamaan.
    zona_mangrove_valid = (
        dekat_air & dekat_darat
    )

    # Pixel Mangrove di luar zona valid.
    perlu_diperbaiki = (
        (original == MANGROVE)
        & (~zona_mangrove_valid)
    )

    jumlah_diperbaiki = int(
        np.count_nonzero(perlu_diperbaiki)
    )

    print(
        f"\nBatas jarak air: {MAX_DISTANCE_WATER_M} meter"
    )
    print(
        f"Batas jarak daratan: {MAX_DISTANCE_LAND_M} meter"
    )
    print(
        "Mangrove yang perlu diperbaiki:",
        f"{jumlah_diperbaiki:,}",
        "pixel"
    )

    # --------------------------------------------------------
    # 4. GANTI KELAS BERDASARKAN TETANGGA
    # --------------------------------------------------------

    print("\n[4/6] Memperbaiki pixel Mangrove...")

    hasil = original.copy()

    # Kelas yang dapat menjadi pengganti Mangrove.
    kelas_pengganti = [
        BUILDING,
        FARMLAND,
        WATER,
        FOREST,
    ]

    if jumlah_diperbaiki > 0:

        jumlah_tetangga = []

        for kode in kelas_pengganti:

            mask = (
                original == kode
            ).astype(np.float32)

            proporsi = uniform_filter(
                mask,
                size=NEIGHBOR_WINDOW,
                mode="nearest"
            )

            jumlah_tetangga.append(proporsi)

        jumlah_tetangga = np.stack(
            jumlah_tetangga,
            axis=0
        )

        # Pilih kelas dengan proporsi tetangga terbesar.
        indeks_terbaik = np.argmax(
            jumlah_tetangga,
            axis=0
        )

        kode_array = np.array(
            kelas_pengganti,
            dtype=np.uint8
        )

        pengganti = kode_array[
            indeks_terbaik
        ]

        hasil[perlu_diperbaiki] = (
            pengganti[perlu_diperbaiki]
        )

        # Jika tidak ada kelas alternatif dalam jendela,
        # cari pixel non-Mangrove terdekat.
        tidak_ada_tetangga = (
            perlu_diperbaiki
            & (
                np.max(jumlah_tetangga, axis=0)
                == 0
            )
        )

        jumlah_fallback = int(
            np.count_nonzero(tidak_ada_tetangga)
        )

        if jumlah_fallback > 0:

            print(
                "Mencari kelas terdekat untuk",
                f"{jumlah_fallback:,} pixel..."
            )

            kelas_valid = np.isin(
                original,
                kelas_pengganti
            )

            indeks_terdekat = distance_transform_edt(
                ~kelas_valid,
                return_distances=False,
                return_indices=True
            )

            kelas_terdekat = original[
                tuple(indeks_terdekat)
            ]

            hasil[tidak_ada_tetangga] = (
                kelas_terdekat[tidak_ada_tetangga]
            )

    else:
        print(
            "Semua pixel Mangrove sudah memenuhi "
            "syarat jarak."
        )

    # Pastikan pixel di luar target tidak berubah.
    hasil[~perlu_diperbaiki] = (
        original[~perlu_diperbaiki]
    )

    # --------------------------------------------------------
    # 5. SIMPAN HASIL
    # --------------------------------------------------------

    print("\n[5/6] Menyimpan hasil...")

    profile.update(
        dtype="uint8",
        count=1,
        compress="lzw"
    )

    with rasterio.open(
        OUTPUT_FILE,
        "w",
        **profile
    ) as dst:
        dst.write(
            hasil.astype(np.uint8),
            1
        )

    print("File tersimpan:", OUTPUT_FILE)

    # --------------------------------------------------------
    # 6. LAPORAN
    # --------------------------------------------------------

    print("\n[6/6] Perbandingan kelas...")

    sebelum = hitung_pixel(original)
    sesudah = hitung_pixel(hasil)

    print("-" * 68)
    print(
        f"{'Kelas':<12}"
        f"{'Sebelum':>15}"
        f"{'Sesudah':>15}"
        f"{'Perubahan':>15}"
    )
    print("-" * 68)

    for nama in CLASS_CODES:

        awal = sebelum[nama]
        akhir = sesudah[nama]
        perubahan = akhir - awal

        print(
            f"{nama:<12}"
            f"{awal:>15,}"
            f"{akhir:>15,}"
            f"{perubahan:>+15,}"
        )

    print("-" * 68)
    print(
        "\nMangrove yang diperbaiki:",
        f"{jumlah_diperbaiki:,}",
        "pixel"
    )
    print("Output:", OUTPUT_FILE)

    print(
        "\nPERHATIAN: Ini filter spasial berbasis jarak, "
        "bukan validasi lapangan. Hasil tetap harus "
        "diperiksa secara visual."
    )

    print("=" * 65)
    print("SELESAI")
    print("=" * 65)


if __name__ == "__main__":
    try:
        main()

    except FileNotFoundError as error:
        print("\nERROR: File tidak ditemukan.")
        print(error)

    except PermissionError as error:
        print("\nERROR: File sedang terkunci atau tidak dapat diakses.")
        print(error)

    except Exception as error:
        print("\nERROR:", type(error).__name__, "-", error)
        raise