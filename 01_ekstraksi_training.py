import os
import glob

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask


# ============================================================
# 1. PENGATURAN
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# GeoJSON TRAINING TERBARU
GEOJSON = os.path.join(
    BASE_DIR,
    "training_jatim_mangrove_diperkecil_clean.geojson"
)

# Output pixel training
OUTPUT = os.path.join(
    BASE_DIR,
    "training_pixel.csv"
)


# ============================================================
# 2. CEK FILE GEOJSON
# ============================================================

if not os.path.exists(GEOJSON):
    raise FileNotFoundError(
        "\nGeoJSON tidak ditemukan:\n"
        + GEOJSON
    )


# ============================================================
# 3. MULAI
# ============================================================

print("=" * 60)
print("EKSTRAKSI PIXEL TRAINING")
print("=" * 60)

print("\nGeoJSON yang digunakan:")
print(
    os.path.basename(GEOJSON)
)


# ============================================================
# 4. CARI FILE TIFF
# ============================================================

print("\nMencari file TIFF...")

tif_files = glob.glob(
    os.path.join(BASE_DIR, "*.tif")
)

tif_files += glob.glob(
    os.path.join(BASE_DIR, "*.tiff")
)


if not tif_files:

    raise FileNotFoundError(
        "Tidak ditemukan file TIFF di folder project."
    )


# ============================================================
# 5. CARI BAND
# ============================================================

BANDS = {}


for file in tif_files:

    nama = os.path.basename(file).upper()


    # B8A harus dicek SEBELUM B08
    if "B8A" in nama:

        if "B8A" not in BANDS:
            BANDS["B8A"] = file


    elif "B08" in nama:

        if "B08" not in BANDS:
            BANDS["B08"] = file


    elif "B04" in nama:

        if "B04" not in BANDS:
            BANDS["B04"] = file


    elif "B11" in nama:

        if "B11" not in BANDS:
            BANDS["B11"] = file


# ============================================================
# 6. CEK BAND
# ============================================================

print("\nBand yang ditemukan:")


for band in [
    "B04",
    "B08",
    "B8A",
    "B11"
]:

    if band not in BANDS:

        raise FileNotFoundError(
            f"\nBand {band} tidak ditemukan."
        )


    print(
        f"{band}: "
        f"{os.path.basename(BANDS[band])}"
    )


# ============================================================
# 7. BACA GEOJSON
# ============================================================

print("\nMembaca GeoJSON...")

gdf = gpd.read_file(
    GEOJSON
)


print(
    f"Jumlah polygon: {len(gdf)}"
)


# ============================================================
# 8. CEK CLASS
# ============================================================

print("\nJumlah setiap kelas:")

print(
    gdf["class"].value_counts()
)


# ============================================================
# 9. CEK CRS
# ============================================================

print("\nCRS GeoJSON:")

print(
    gdf.crs
)


# ============================================================
# 10. BUKA RASTER
# ============================================================

print("\nMembuka raster...")

srcs = {}


for band, path in BANDS.items():

    src = rasterio.open(path)

    srcs[band] = src


    print(
        f"{band}: "
        f"{src.width} x {src.height} | "
        f"CRS = {src.crs}"
    )


# ============================================================
# 11. CEK CRS RASTER
# ============================================================

raster_crs = srcs["B04"].crs


if gdf.crs != raster_crs:

    print("\nCRS berbeda.")

    print(
        "Mengubah CRS GeoJSON "
        "agar sama dengan raster..."
    )


    gdf = gdf.to_crs(
        raster_crs
    )


else:

    print(
        "\nCRS GeoJSON dan raster sudah sama."
    )


# ============================================================
# 12. EKSTRAKSI PIXEL
# ============================================================

print("\nMulai ekstraksi pixel...")

print("-" * 60)


hasil = []


berhasil = 0
gagal = 0


total_polygon = len(gdf)


for idx, row in gdf.iterrows():

    polygon = [
        row.geometry
    ]

    kelas = row["class"]

    nomor = idx + 1


    print(
        f"Polygon "
        f"{nomor:02d}/{total_polygon} | "
        f"{kelas}"
    )


    try:

        # ====================================================
        # B04
        # ====================================================

        data_b04, _ = mask(
            srcs["B04"],
            polygon,
            crop=True,
            filled=False
        )


        # ====================================================
        # B08
        # ====================================================

        data_b08, _ = mask(
            srcs["B08"],
            polygon,
            crop=True,
            filled=False
        )


        # ====================================================
        # B8A
        # ====================================================

        data_b8a, _ = mask(
            srcs["B8A"],
            polygon,
            crop=True,
            filled=False
        )


        # ====================================================
        # B11
        # ====================================================

        data_b11, _ = mask(
            srcs["B11"],
            polygon,
            crop=True,
            filled=False
        )


        # ====================================================
        # AMBIL BAND PERTAMA
        # ====================================================

        b04 = data_b04[0]
        b08 = data_b08[0]
        b8a = data_b8a[0]
        b11 = data_b11[0]


        # ====================================================
        # SAMAKAN UKURAN ARRAY
        # ====================================================

        tinggi = min(
            b04.shape[0],
            b08.shape[0],
            b8a.shape[0],
            b11.shape[0]
        )


        lebar = min(
            b04.shape[1],
            b08.shape[1],
            b8a.shape[1],
            b11.shape[1]
        )


        b04 = b04[
            :tinggi,
            :lebar
        ]


        b08 = b08[
            :tinggi,
            :lebar
        ]


        b8a = b8a[
            :tinggi,
            :lebar
        ]


        b11 = b11[
            :tinggi,
            :lebar
        ]


        # ====================================================
        # PIXEL VALID DI SEMUA BAND
        # ====================================================

        valid = (
            (~b04.mask)
            &
            (~b08.mask)
            &
            (~b8a.mask)
            &
            (~b11.mask)
        )


        nilai_b04 = b04.data[
            valid
        ]


        nilai_b08 = b08.data[
            valid
        ]


        nilai_b8a = b8a.data[
            valid
        ]


        nilai_b11 = b11.data[
            valid
        ]


        jumlah_pixel = len(
            nilai_b04
        )


        # ====================================================
        # CEK PIXEL
        # ====================================================

        if jumlah_pixel == 0:

            print(
                "   GAGAL -> "
                "tidak ada pixel valid"
            )

            gagal += 1

            continue


        # ====================================================
        # SIMPAN PIXEL
        # ====================================================

        for i in range(
            jumlah_pixel
        ):

            hasil.append({

                "polygon_id": nomor,

                "class": kelas,

                "B04": float(
                    nilai_b04[i]
                ),

                "B08": float(
                    nilai_b08[i]
                ),

                "B8A": float(
                    nilai_b8a[i]
                ),

                "B11": float(
                    nilai_b11[i]
                )

            })


        print(
            f"   OK -> "
            f"{jumlah_pixel} pixel"
        )


        berhasil += 1


    except ValueError as e:

        if (
            "do not overlap raster"
            in str(e)
        ):

            print(
                "   GAGAL -> "
                "polygon tidak overlap raster"
            )

        else:

            print(
                f"   GAGAL -> {e}"
            )


        gagal += 1


    except Exception as e:

        print(
            f"   GAGAL -> {e}"
        )

        gagal += 1


# ============================================================
# 13. TUTUP RASTER
# ============================================================

for src in srcs.values():

    src.close()


# ============================================================
# 14. CEK HASIL
# ============================================================

print(
    "\n" + "=" * 60
)


if not hasil:

    raise RuntimeError(
        "Tidak ada pixel training "
        "yang berhasil diekstrak."
    )


# ============================================================
# 15. BUAT DATAFRAME
# ============================================================

df = pd.DataFrame(
    hasil
)


# ============================================================
# 16. SIMPAN CSV
# ============================================================

df.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# 17. HASIL AKHIR
# ============================================================

print(
    "EKSTRAKSI SELESAI!"
)


print(
    f"\nPolygon berhasil : {berhasil}"
)


print(
    f"Polygon gagal    : {gagal}"
)


print(
    f"Total pixel      : {len(df)}"
)


# ============================================================
# 18. JUMLAH PIXEL PER KELAS
# ============================================================

print(
    "\nJumlah pixel per kelas:"
)


print(
    df["class"].value_counts()
)


# ============================================================
# 19. OUTPUT
# ============================================================

print(
    "\nFile output:"
)


print(
    OUTPUT
)


# ============================================================
# 20. KOLOM
# ============================================================

print(
    "\nKolom:"
)


print(
    list(df.columns)
)


print(
    "=" * 60
)