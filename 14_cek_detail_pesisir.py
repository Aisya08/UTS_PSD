import numpy as np
import rasterio
import matplotlib.pyplot as plt

from glob import glob
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle, Patch


# ============================================================
# NAMA FILE
# ============================================================

FILE_ASLI = "hasil_klasifikasi_jatim.tif"

FILE_FILTER = (
    "hasil_klasifikasi_jatim_mangrove_koridor_pesisir.tif"
)

OUTPUT_GAMBAR = "detail_pemeriksaan_pesisir.png"


# ============================================================
# KODE DAN WARNA KELAS
# ============================================================

CLASS_CODES = {
    "Building": 1,
    "Farmland": 2,
    "Water": 3,
    "Forest": 4,
    "Mangrove": 5,
}

COLORS = [
    "#FFFFFF",  # 0 = NoData
    "#E53935",  # 1 = Building
    "#FDD835",  # 2 = Farmland
    "#1E88E5",  # 3 = Water
    "#43A047",  # 4 = Forest
    "#8E24AA",  # 5 = Mangrove
]

CLASS_CMAP = ListedColormap(COLORS)

CLASS_NORM = BoundaryNorm(
    np.arange(-0.5, 6.5, 1),
    CLASS_CMAP.N
)


# ============================================================
# CARI BAND
# ============================================================

def cari_band(nama):
    pola = [
        f"*Sentinel-2_L2A_{nama}_(Raw).tiff",
        f"*Sentinel-2_L2A_{nama}_(Raw).tif",
    ]

    for pattern in pola:
        files = sorted(glob(pattern))

        if files:
            return files[0]

    raise FileNotFoundError(
        f"Band {nama} tidak ditemukan."
    )


# ============================================================
# NORMALISASI UNTUK TAMPILAN
# ============================================================

def normalisasi(data):
    data = data.astype(np.float32)

    valid = np.isfinite(data)

    if not np.any(valid):
        return np.zeros_like(data)

    bawah, atas = np.percentile(
        data[valid],
        [2, 98]
    )

    if atas <= bawah:
        return np.zeros_like(data)

    hasil = (data - bawah) / (atas - bawah)

    return np.clip(hasil, 0, 1)


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():

    print("=" * 60)
    print("PEMERIKSAAN DETAIL MANGROVE DI PESISIR")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. BACA KLASIFIKASI
    # --------------------------------------------------------

    print("\nMembaca klasifikasi asli dan hasil filter...")

    with rasterio.open(FILE_ASLI) as src:
        asli = src.read(1)
        profile_asli = src.profile.copy()

    with rasterio.open(FILE_FILTER) as src:
        hasil = src.read(1)
        profile_hasil = src.profile.copy()

    if asli.shape != hasil.shape:
        raise ValueError(
            "Ukuran raster asli dan hasil filter berbeda."
        )

    if (
        profile_asli["crs"] != profile_hasil["crs"]
        or profile_asli["transform"] != profile_hasil["transform"]
    ):
        raise ValueError(
            "Posisi raster asli dan hasil filter tidak sama."
        )

    tinggi, lebar = asli.shape

    print("Ukuran raster:", lebar, "x", tinggi)

    # --------------------------------------------------------
    # 2. BACA BAND UNTUK CITRA FALSE COLOR
    # --------------------------------------------------------

    print("\nMembaca band B11, B08, dan B04...")

    bands = {}

    for nama in ["B11", "B08", "B04"]:
        path = cari_band(nama)

        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)

            if data.shape != asli.shape:
                raise ValueError(
                    f"Ukuran band {nama} tidak cocok."
                )

            if (
                src.crs != profile_asli["crs"]
                or src.transform != profile_asli["transform"]
            ):
                raise ValueError(
                    f"Posisi band {nama} tidak cocok."
                )

        bands[nama] = data

        print(f"{nama}: {path}")

    # Komposisi false color: R=B11, G=B08, B=B04.
    false_color = np.dstack([
        normalisasi(bands["B11"]),
        normalisasi(bands["B08"]),
        normalisasi(bands["B04"]),
    ])

    # --------------------------------------------------------
    # 3. PILIH POTONGAN PESISIR SEBAGAI DETAIL
    # --------------------------------------------------------

    # Pesisir berada di sisi kanan citra.
    # Kita ambil 35% sisi kanan dan 75% bagian tengah
    # agar garis pantai terlihat lebih besar.
    row_start = int(tinggi * 0.10)
    row_end = int(tinggi * 0.85)

    col_start = int(lebar * 0.65)
    col_end = lebar

    crop_rows = slice(row_start, row_end)
    crop_cols = slice(col_start, col_end)

    print("\nArea detail yang diperbesar:")
    print("Baris :", row_start, "sampai", row_end)
    print("Kolom :", col_start, "sampai", col_end)

    # --------------------------------------------------------
    # 4. GAMBAR OVERVIEW DAN DETAIL PESISIR
    # --------------------------------------------------------

    print("\nMembuat visualisasi...")

    fig, ax = plt.subplots(
        2,
        3,
        figsize=(18, 11)
    )

    # Baris atas: overview.
    ax[0, 0].imshow(false_color)
    ax[0, 0].set_title("Overview Sentinel-2 False Color")

    # Tandai area crop detail dengan kotak merah.
    kotak = Rectangle(
        (col_start, row_start),
        col_end - col_start,
        row_end - row_start,
        edgecolor="red",
        facecolor="none",
        linewidth=2
    )

    ax[0, 0].add_patch(kotak)
    ax[0, 0].axis("off")

    ax[0, 1].imshow(
        asli,
        cmap=CLASS_CMAP,
        norm=CLASS_NORM,
        interpolation="nearest"
    )

    ax[0, 1].set_title("Overview Klasifikasi Asli")
    ax[0, 1].axis("off")

    ax[0, 2].imshow(
        hasil,
        cmap=CLASS_CMAP,
        norm=CLASS_NORM,
        interpolation="nearest"
    )

    ax[0, 2].set_title("Overview Hasil Filter")
    ax[0, 2].axis("off")

    # Baris bawah: detail crop.
    ax[1, 0].imshow(
        false_color[crop_rows, crop_cols]
    )

    ax[1, 0].set_title("DETAIL: Citra Satelit")

    ax[1, 1].imshow(
        asli[crop_rows, crop_cols],
        cmap=CLASS_CMAP,
        norm=CLASS_NORM,
        interpolation="nearest"
    )

    ax[1, 1].set_title("DETAIL: Klasifikasi Asli")

    ax[1, 2].imshow(
        hasil[crop_rows, crop_cols],
        cmap=CLASS_CMAP,
        norm=CLASS_NORM,
        interpolation="nearest"
    )

    ax[1, 2].set_title("DETAIL: Hasil Filter")

    for row in range(2):
        for col in range(3):
            ax[row, col].axis("off")

    legenda = [
        Patch(facecolor=COLORS[1], label="Building"),
        Patch(facecolor=COLORS[2], label="Farmland"),
        Patch(facecolor=COLORS[3], label="Water"),
        Patch(facecolor=COLORS[4], label="Forest"),
        Patch(facecolor=COLORS[5], label="Mangrove"),
    ]

    fig.legend(
        handles=legenda,
        loc="lower center",
        ncol=5,
        frameon=False
    )

    fig.suptitle(
        "Pemeriksaan Detail Mangrove di Kawasan Pesisir",
        fontsize=15
    )

    plt.tight_layout(
        rect=[0, 0.07, 1, 0.95]
    )

    plt.savefig(
        OUTPUT_GAMBAR,
        dpi=180,
        bbox_inches="tight"
    )

    print("Gambar disimpan:", OUTPUT_GAMBAR)

    plt.show()

    print("\nPemeriksaan selesai.")


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