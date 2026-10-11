import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


# ============================================================
# KONFIGURASI FILE
# ============================================================

FILE_ASLI = "hasil_klasifikasi_jatim.tif"

FILE_PERBAIKAN = (
    "hasil_klasifikasi_jatim_mangrove_diperbaiki.tif"
)

OUTPUT_GAMBAR = "perbandingan_mangrove_ndvi.png"


# ============================================================
# MENCARI FILE BAND SENTINEL-2
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
        f"File band {nama_band} tidak ditemukan. "
        "Pastikan script berada di folder TIFF Sentinel-2."
    )


# ============================================================
# NORMALISASI BAND UNTUK VISUALISASI
# ============================================================

def normalisasi(band):
    band = band.astype(np.float32)

    valid = np.isfinite(band)

    if not np.any(valid):
        return np.zeros_like(band, dtype=np.float32)

    nilai_valid = band[valid]

    batas_bawah, batas_atas = np.percentile(
        nilai_valid,
        [2, 98]
    )

    if batas_atas <= batas_bawah:
        return np.zeros_like(band, dtype=np.float32)

    hasil = (
        (band - batas_bawah)
        / (batas_atas - batas_bawah)
    )

    return np.clip(hasil, 0, 1)


# ============================================================
# MEMBUAT CITRA FALSE COLOR B11-B08-B04
# ============================================================

def buat_false_color():

    print("\nMencari band Sentinel-2...")

    file_b11 = cari_band("B11")
    file_b08 = cari_band("B08")
    file_b04 = cari_band("B04")

    print("B11:", file_b11)
    print("B08:", file_b08)
    print("B04:", file_b04)

    with rasterio.open(file_b11) as src:
        b11 = src.read(1)

    with rasterio.open(file_b08) as src:
        b08 = src.read(1)

    with rasterio.open(file_b04) as src:
        b04 = src.read(1)

    if not (
        b11.shape == b08.shape == b04.shape
    ):
        raise ValueError(
            "Ukuran band B11, B08, dan B04 berbeda."
        )

    # Komposisi RGB:
    # Merah = B11
    # Hijau = B08
    # Biru = B04
    rgb = np.dstack([
        normalisasi(b11),
        normalisasi(b08),
        normalisasi(b04),
    ])

    return rgb


# ============================================================
# MEMBACA RASTER KLASIFIKASI
# ============================================================

def baca_klasifikasi(nama_file):

    with rasterio.open(nama_file) as src:
        data = src.read(1)
        profile = src.profile.copy()

    return data, profile


# ============================================================
# WARNA KELAS KLASIFIKASI
# ============================================================

warna_kelas = [
    "#FFFFFF",  # 0 = NoData
    "#E53935",  # 1 = Building
    "#FDD835",  # 2 = Farmland
    "#1E88E5",  # 3 = Water
    "#43A047",  # 4 = Forest
    "#8E24AA",  # 5 = Mangrove
]

cmap = ListedColormap(warna_kelas)

norm = BoundaryNorm(
    np.arange(-0.5, 6.5, 1),
    cmap.N
)


# ============================================================
# MENAMPILKAN PERBANDINGAN
# ============================================================

def main():

    print("=" * 65)
    print("PERBANDINGAN HASIL FILTER MANGROVE BERDASARKAN NDVI")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. MEMBACA DATA
    # --------------------------------------------------------

    print("\n[1/4] Membaca citra false color...")

    rgb = buat_false_color()

    print("\n[2/4] Membaca klasifikasi asli...")

    klasifikasi_asli, profil_asli = baca_klasifikasi(
        FILE_ASLI
    )

    print("\n[3/4] Membaca hasil filter NDVI...")

    klasifikasi_ndvi, profil_ndvi = baca_klasifikasi(
        FILE_PERBAIKAN
    )

    # --------------------------------------------------------
    # 2. MEMERIKSA KESESUAIAN UKURAN DAN POSISI RASTER
    # --------------------------------------------------------

    if klasifikasi_asli.shape != klasifikasi_ndvi.shape:
        raise ValueError(
            "Ukuran raster asli dan hasil NDVI berbeda."
        )

    if rgb.shape[:2] != klasifikasi_asli.shape:
        raise ValueError(
            "Ukuran citra false color berbeda dari klasifikasi."
        )

    if (
        profil_asli["transform"]
        != profil_ndvi["transform"]
    ):
        raise ValueError(
            "Posisi pixel raster asli dan hasil NDVI berbeda."
        )

    if profil_asli["crs"] != profil_ndvi["crs"]:
        raise ValueError(
            "CRS raster asli dan hasil NDVI berbeda."
        )

    print("\nSemua raster berhasil diperiksa.")
    print("Ukuran raster:", klasifikasi_asli.shape)
    print("CRS:", profil_asli["crs"])

    # --------------------------------------------------------
    # 3. HITUNG JUMLAH PIXEL SETIAP KELAS
    # --------------------------------------------------------

    daftar_kelas = [
        ("Building", 1),
        ("Farmland", 2),
        ("Water", 3),
        ("Forest", 4),
        ("Mangrove", 5),
    ]

    print("\nJumlah pixel setiap kelas:")
    print("-" * 57)
    print(
        f"{'Kelas':<12}"
        f"{'Klasifikasi Asli':>22}"
        f"{'Filter NDVI':>20}"
    )
    print("-" * 57)

    for nama, kode in daftar_kelas:

        jumlah_asli = int(
            np.count_nonzero(
                klasifikasi_asli == kode
            )
        )

        jumlah_ndvi = int(
            np.count_nonzero(
                klasifikasi_ndvi == kode
            )
        )

        print(
            f"{nama:<12}"
            f"{jumlah_asli:>22,}"
            f"{jumlah_ndvi:>20,}"
        )

    print("-" * 57)

    # --------------------------------------------------------
    # 4. MEMBUAT GAMBAR TIGA PANEL
    # --------------------------------------------------------

    print("\n[4/4] Membuat gambar perbandingan...")

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(20, 8)
    )

    # Panel pertama: citra Sentinel-2.
    ax[0].imshow(rgb)

    ax[0].set_title(
        "Citra Sentinel-2 False Color\nB11 - B08 - B04",
        fontsize=12
    )

    ax[0].axis("off")

    # Panel kedua: klasifikasi asli.
    ax[1].imshow(
        klasifikasi_asli,
        cmap=cmap,
        norm=norm,
        interpolation="nearest"
    )

    ax[1].set_title(
        "Klasifikasi Asli\nSebelum Filter NDVI",
        fontsize=12
    )

    ax[1].axis("off")

    # Panel ketiga: klasifikasi setelah filter NDVI.
    ax[2].imshow(
        klasifikasi_ndvi,
        cmap=cmap,
        norm=norm,
        interpolation="nearest"
    )

    ax[2].set_title(
        "Klasifikasi Setelah Filter NDVI",
        fontsize=12
    )

    ax[2].axis("off")

    # --------------------------------------------------------
    # LEGENDA
    # --------------------------------------------------------

    legenda = [
        Patch(
            facecolor="#E53935",
            label="Building"
        ),
        Patch(
            facecolor="#FDD835",
            label="Farmland"
        ),
        Patch(
            facecolor="#1E88E5",
            label="Water"
        ),
        Patch(
            facecolor="#43A047",
            label="Forest"
        ),
        Patch(
            facecolor="#8E24AA",
            label="Mangrove"
        ),
    ]

    fig.legend(
        handles=legenda,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.04),
        fontsize=10
    )

    fig.suptitle(
        "Evaluasi Perbaikan Klasifikasi Mangrove",
        fontsize=15,
        y=0.97
    )

    plt.tight_layout(
        rect=[0, 0.10, 1, 0.91]
    )

    # Simpan gambar untuk laporan tugas.
    plt.savefig(
        OUTPUT_GAMBAR,
        dpi=200,
        bbox_inches="tight"
    )

    print("\nGambar berhasil disimpan:")
    print(OUTPUT_GAMBAR)

    # Tampilkan di layar.
    plt.show()

    print("\nPerbandingan selesai.")


# ============================================================
# MENJALANKAN PROGRAM
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except FileNotFoundError as error:
        print("\nERROR: File tidak ditemukan.")
        print(error)

    except Exception as error:
        print("\nERROR:", type(error).__name__)
        print(error)
        raise