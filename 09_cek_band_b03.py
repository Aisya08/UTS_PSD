import glob
import rasterio


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


print("=" * 60)
print("PEMERIKSAAN BAND B03 DAN B11")
print("=" * 60)

data = {}

for band in ["B03", "B11"]:
    path = cari_band(band)

    with rasterio.open(path) as src:
        data[band] = {
            "path": path,
            "shape": src.shape,
            "crs": src.crs,
            "transform": src.transform,
            "dtype": src.dtypes[0],
            "bounds": src.bounds,
        }

        print(f"\nBand {band}")
        print("File      :", path)
        print("Ukuran    :", src.width, "x", src.height)
        print("CRS       :", src.crs)
        print("Tipe data :", src.dtypes[0])
        print("Transform :", src.transform)

print("\n" + "=" * 60)
print("PEMERIKSAAN KESESUAIAN")
print("=" * 60)

ukuran_sama = data["B03"]["shape"] == data["B11"]["shape"]
crs_sama = data["B03"]["crs"] == data["B11"]["crs"]
transform_sama = (
    data["B03"]["transform"]
    == data["B11"]["transform"]
)

print("Ukuran sama    :", ukuran_sama)
print("CRS sama       :", crs_sama)
print("Transform sama :", transform_sama)

if ukuran_sama and crs_sama and transform_sama:
    print("\nHASIL: Kedua band sejajar dan siap diperiksa.")
else:
    print(
        "\nHASIL: Kedua band belum cocok sepenuhnya. "
        "Perlu penyelarasan sebelum menghitung indeks air."
    )

print("=" * 60)