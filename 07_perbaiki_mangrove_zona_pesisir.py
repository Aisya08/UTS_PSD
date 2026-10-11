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
    "hasil_klasifikasi_jatim_mangrove_zona_pesisir.tif"
)

# Kode kelas
BUILDING = 1
FARMLAND = 2
WATER = 3
FOREST = 4
MANGROVE = 5

# Batas awal zona pesisir
MAX_DISTANCE_WATER_M = 1000

# Ukuran jendela untuk mencari kelas pengganti
NEIGHBOR_WINDOW = 21

CLASS_NAMES = {
    1: "Building",
    2: "Farmland",
    3: "Water",
    4: "Forest",
    5: "Mangrove",
}


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():

    print("=" * 65)
    print("FILTER MANGROVE BERDASARKAN ZONA PESISIR")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. BACA KLASIFIKASI ASLI
    # --------------------------------------------------------

    print("\n[1/5] Membaca klasifikasi asli...")

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

    print("Ukuran raster:", original.shape)
    print("CRS:", crs)

    mangrove_awal = original == MANGROVE
    water_mask = original == WATER

    jumlah_mangrove_awal = int(
        np.count_nonzero(mangrove_awal)
    )

    print(
        "Mangrove awal:",
        f"{jumlah_mangrove_awal:,}",
        "pixel"
    )

    if not np.any(mangrove_awal):
        raise ValueError("Tidak ditemukan kelas Mangrove.")

    if not np.any(water_mask):
        raise ValueError("Tidak ditemukan kelas Water.")

    # --------------------------------------------------------
    # 2. CARI KOMPONEN AIR TERBESAR
    # --------------------------------------------------------

    print("\n[2/5] Mengidentifikasi area air terbesar...")

    struktur = np.ones((3, 3), dtype=np.uint8)

    label_air, jumlah_komponen = label(
        water_mask,
        structure=struktur,
    )

    ukuran = np.bincount(label_air.ravel())
    ukuran[0] = 0

    label_terbesar = int(np.argmax(ukuran))
    air_terbesar = label_air == label_terbesar

    print("Jumlah komponen air:", jumlah_komponen)
    print(
        "Pixel komponen terbesar:",
        f"{np.count_nonzero(air_terbesar):,}"
    )

    print(
        "Periksa secara visual apakah komponen terbesar "
        "benar-benar mewakili laut."
    )

    # --------------------------------------------------------
    # 3. HITUNG JARAK KE AIR
    # --------------------------------------------------------

    print("\n[3/5] Menghitung jarak dari area air...")

    latitude_tengah = (
        bounds.top + bounds.bottom
    ) / 2

    pixel_y_m = abs(transform.e) * 111320.0

    pixel_x_m = (
        abs(transform.a)
        * 111320.0
        * np.cos(np.deg2rad(latitude_tengah))
    )

    if pixel_x_m <= 0 or pixel_y_m <= 0:
        raise ValueError("Resolusi raster tidak valid.")

    jarak_air = distance_transform_edt(
        ~air_terbesar,
        sampling=(pixel_y_m, pixel_x_m),
    )

    # Pertahankan Mangrove di dalam zona pesisir.
    zona_pesisir = (
        jarak_air <= MAX_DISTANCE_WATER_M
    )

    perlu_diperbaiki = (
        mangrove_awal
        & (~zona_pesisir)
    )

    jumlah_diperbaiki = int(
        np.count_nonzero(perlu_diperbaiki)
    )

    jumlah_dipertahankan = int(
        np.count_nonzero(
            mangrove_awal & zona_pesisir
        )
    )

    print(
        "Batas jarak air:",
        MAX_DISTANCE_WATER_M,
        "meter"
    )

    print(
        "Mangrove dipertahankan:",
        f"{jumlah_dipertahankan:,}",
        "pixel"
    )

    print(
        "Mangrove di luar zona:",
        f"{jumlah_diperbaiki:,}",
        "pixel"
    )

    # --------------------------------------------------------
    # 4. GANTI KELAS MANGROVE DI LUAR ZONA
    # --------------------------------------------------------

    print("\n[4/5] Memperbaiki pixel di luar zona pesisir...")

    hasil = original.copy()

    kelas_alternatif = [
        BUILDING,
        FARMLAND,
        WATER,
        FOREST,
    ]

    if jumlah_diperbaiki > 0:

        frekuensi_tetangga = np.stack([
            uniform_filter(
                (original == kode).astype(np.float32),
                size=NEIGHBOR_WINDOW,
                mode="nearest",
            )
            for kode in kelas_alternatif
        ])

        indeks_terbaik = np.argmax(
            frekuensi_tetangga,
            axis=0
        )

        kode_pengganti = np.asarray(
            kelas_alternatif,
            dtype=np.uint8
        )[indeks_terbaik]

        hasil[perlu_diperbaiki] = (
            kode_pengganti[perlu_diperbaiki]
        )

        # Fallback jika tidak ada kelas alternatif
        # di sekitar pixel yang akan diganti.
        tidak_ada_tetangga = (
            perlu_diperbaiki
            & (
                np.max(frekuensi_tetangga, axis=0)
                == 0
            )
        )

        if np.any(tidak_ada_tetangga):

            kelas_valid = np.isin(
                original,
                kelas_alternatif
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

    # Pastikan pixel yang tidak perlu diperbaiki
    # tetap sama dengan klasifikasi asli.
    hasil[~perlu_diperbaiki] = (
        original[~perlu_diperbaiki]
    )

    # --------------------------------------------------------
    # 5. SIMPAN DAN CETAK PERBANDINGAN
    # --------------------------------------------------------

    print("\n[5/5] Menyimpan hasil...")

    profile.update(
        dtype="uint8",
        count=1,
        compress="lzw",
    )

    with rasterio.open(
        OUTPUT_FILE,
        "w",
        **profile
    ) as dst:
        dst.write(hasil.astype(np.uint8), 1)

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

        sebelum = int(
            np.count_nonzero(original == kode)
        )

        sesudah = int(
            np.count_nonzero(hasil == kode)
        )

        print(
            f"{nama:<12}"
            f"{sebelum:>15,}"
            f"{sesudah:>15,}"
            f"{sesudah-sebelum:>+15,}"
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
            error
        )
        raise