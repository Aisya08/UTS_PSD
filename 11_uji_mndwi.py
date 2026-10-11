import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm


# ============================================================
# KONFIGURASI
# ============================================================

CLASS_FILE = "hasil_klasifikasi_jatim.tif"

OUTPUT_IMAGE = "uji_mndwi_jatim.png"

BAND_NAMES = ["B03", "B04", "B08", "B11"]

CLASS_NAMES = {
    1: "Building",
    2: "Farmland",
    3: "Water",
    4: "Forest",
    5: "Mangrove",
}


# ============================================================
# CARI FILE BAND
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
# BACA BAND DAN PERIKSA KESESUAIAN
# ============================================================

def baca_band(nama_band):
    path = cari_band(nama_band)

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    print(f"{nama_band}: {path}")

    return data, profile


# ============================================================
# NORMALISASI UNTUK GAMBAR FALSE COLOR
# ============================================================

def normalisasi(band):
    valid = np.isfinite(band)

    if not np.any(valid):
        return np.zeros_like(band)

    nilai = band[valid]

    bawah, atas = np.percentile(nilai, [2, 98])

    if atas <= bawah:
        return np.zeros_like(band)

    hasil = (band - bawah) / (atas - bawah)

    return np.clip(hasil, 0, 1)


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():

    print("=" * 65)
    print("UJI MNDWI UNTUK DETEKSI AREA AIR")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. BACA BAND
    # --------------------------------------------------------

    print("\n[1/5] Membaca band Sentinel-2...")

    bands = {}

    for nama in BAND_NAMES:
        bands[nama], profil = baca_band(nama)

        if nama == "B03":
            profil_referensi = profil
        else:
            if bands[nama].shape != bands["B03"].shape:
                raise ValueError(
                    f"Ukuran band {nama} berbeda dari B03."
                )

            with rasterio.open(cari_band(nama)) as src:
                if (
                    src.crs != profil_referensi["crs"]
                    or src.transform
                    != profil_referensi["transform"]
                ):
                    raise ValueError(
                        f"CRS atau transform band {nama} "
                        "berbeda dari B03."
                    )

    # --------------------------------------------------------
    # 2. BACA KLASIFIKASI LAMA
    # --------------------------------------------------------

    print("\n[2/5] Membaca klasifikasi lama...")

    with rasterio.open(CLASS_FILE) as src:
        klasifikasi = src.read(1)
        class_profile = src.profile.copy()

    if klasifikasi.shape != bands["B03"].shape:
        raise ValueError(
            "Ukuran klasifikasi tidak cocok dengan band."
        )

    if (
        class_profile["crs"] != profil_referensi["crs"]
        or class_profile["transform"]
        != profil_referensi["transform"]
    ):
        raise ValueError(
            "Posisi raster klasifikasi tidak cocok "
            "dengan band Sentinel-2."
        )

    print("Ukuran raster:", klasifikasi.shape)
    print("CRS:", class_profile["crs"])

    # --------------------------------------------------------
    # 3. HITUNG MNDWI
    # --------------------------------------------------------

    print("\n[3/5] Menghitung MNDWI...")

    green = bands["B03"]
    swir = bands["B11"]

    penyebut = green + swir

    valid = (
        np.isfinite(green)
        & np.isfinite(swir)
        & (green >= 0)
        & (swir >= 0)
        & (penyebut > 0)
    )

    mndwi = np.full(
        green.shape,
        np.nan,
        dtype=np.float32
    )

    np.divide(
        green - swir,
        penyebut,
        out=mndwi,
        where=valid
    )

    nilai_valid = mndwi[np.isfinite(mndwi)]

    if nilai_valid.size == 0:
        raise ValueError(
            "Tidak ditemukan nilai MNDWI yang valid."
        )

    print("Jumlah pixel valid:", f"{nilai_valid.size:,}")
    print("Minimum MNDWI:", round(float(np.min(nilai_valid)), 3))
    print("Median MNDWI:", round(float(np.median(nilai_valid)), 3))
    print("Maksimum MNDWI:", round(float(np.max(nilai_valid)), 3))

    # --------------------------------------------------------
    # 4. UJI BEBERAPA AMBANG AIR
    # --------------------------------------------------------

    print("\n[4/5] Menguji kandidat area air...")

    mangrove = klasifikasi == 5

    for threshold in [0.0, 0.1, 0.2]:

        mask_air = (
            np.isfinite(mndwi)
            & (mndwi > threshold)
        )

        jumlah_air = int(np.count_nonzero(mask_air))

        jumlah_mangrove_dalam_air = int(
            np.count_nonzero(
                mask_air & mangrove
            )
        )

        print(f"\nAmbang MNDWI > {threshold:.1f}")
        print("Kandidat air:", f"{jumlah_air:,}", "pixel")
        print(
            "Pixel Mangrove yang bertumpang tindih:",
            f"{jumlah_mangrove_dalam_air:,}"
        )

    # Ambang awal untuk divisualisasikan.
    threshold_visual = 0.0

    mask_air = (
        np.isfinite(mndwi)
        & (mndwi > threshold_visual)
    )

    # --------------------------------------------------------
    # 5. BUAT GAMBAR DIAGNOSTIK
    # --------------------------------------------------------

    print("\n[5/5] Membuat gambar diagnostik...")

    # False color: R=B11, G=B08, B=B04.
    false_color = np.dstack([
        normalisasi(bands["B11"]),
        normalisasi(bands["B08"]),
        normalisasi(bands["B04"]),
    ])

    cmap_kelas = ListedColormap([
        "#FFFFFF",  # 0 NoData
        "#E53935",  # 1 Building
        "#FDD835",  # 2 Farmland
        "#1E88E5",  # 3 Water
        "#43A047",  # 4 Forest
        "#8E24AA",  # 5 Mangrove
    ])

    norm_kelas = BoundaryNorm(
        np.arange(-0.5, 6.5, 1),
        cmap_kelas.N
    )

    cmap_air = ListedColormap([
        "#D9D9D9",  # MNDWI <= ambang
        "#168BFF",  # MNDWI > ambang
    ])

    fig, axes = plt.subplots(
        2, 2,
        figsize=(15, 12)
    )

    # Panel 1: citra satelit.
    axes[0, 0].imshow(false_color)
    axes[0, 0].set_title(
        "Sentinel-2 False Color B11-B08-B04"
    )
    axes[0, 0].axis("off")

    # Panel 2: nilai MNDWI.
    gambar_mndwi = axes[0, 1].imshow(
        np.ma.masked_invalid(mndwi),
        cmap="BrBG",
        vmin=-0.5,
        vmax=0.5
    )

    axes[0, 1].set_title("MNDWI (B03 dan B11)")
    axes[0, 1].axis("off")

    fig.colorbar(
        gambar_mndwi,
        ax=axes[0, 1],
        fraction=0.046,
        pad=0.04
    )

    # Panel 3: area air kandidat.
    axes[1, 0].imshow(
        mask_air,
        cmap=cmap_air,
        vmin=0,
        vmax=1
    )

    axes[1, 0].set_title(
        "Kandidat Air dari MNDWI > 0"
    )
    axes[1, 0].axis("off")

    # Panel 4: klasifikasi lama.
    axes[1, 1].imshow(
        klasifikasi,
        cmap=cmap_kelas,
        norm=norm_kelas,
        interpolation="nearest"
    )

    axes[1, 1].set_title(
        "Klasifikasi Lama"
    )
    axes[1, 1].axis("off")

    fig.suptitle(
        "Pemeriksaan MNDWI Sebelum Perbaikan Mangrove",
        fontsize=15
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_IMAGE,
        dpi=180,
        bbox_inches="tight"
    )

    print("\nGambar tersimpan:", OUTPUT_IMAGE)

    plt.show()

    print("\nUji MNDWI selesai.")
    print(
        "Belum ada kelas pada raster klasifikasi "
        "yang diubah oleh script ini."
    )


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