import glob
import numpy as np
import rasterio


BANDS = ["B03", "B04", "B08", "B8A", "B11"]


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


def periksa_band(nama_band):
    path = cari_band(nama_band)

    with rasterio.open(path) as src:
        data = src.read(1, masked=True)
        values = np.asarray(data.compressed())
        values = values[np.isfinite(values)]

        print(f"\n{'=' * 60}")
        print(f"BAND {nama_band}")
        print("=" * 60)

        print("File         :", path)
        print("Data type    :", src.dtypes[0])
        print("Ukuran       :", src.width, "x", src.height)
        print("CRS          :", src.crs)
        print("NoData       :", src.nodata)
        print("Scale        :", src.scales)
        print("Offset       :", src.offsets)

        print("\nMetadata band:")
        print(src.tags(1))

        print("\nMetadata file:")
        print(src.tags())

        if values.size == 0:
            print("Tidak ada nilai valid.")
            return

        print("\nStatistik semua nilai valid:")
        print("Jumlah       :", f"{values.size:,}")
        print("Minimum      :", float(np.min(values)))
        print("Maksimum     :", float(np.max(values)))
        print("Persentil 2  :", float(np.percentile(values, 2)))
        print("Median       :", float(np.median(values)))
        print("Persentil 98 :", float(np.percentile(values, 98)))

        if np.issubdtype(values.dtype, np.integer):
            print("\nPemeriksaan nilai khusus:")
            print("Jumlah nilai 0     :", int(np.count_nonzero(values == 0)))
            print("Jumlah nilai 65535 :", int(np.count_nonzero(values == 65535)))

        # Statistik tambahan dengan mengecualikan nilai ekstrem
        # 0 dan 65535 agar perbandingan lebih mudah diperiksa.
        bersih = values[
            (values > 0) & (values < 65535)
        ]

        print("\nStatistik tanpa nilai 0 dan 65535:")
        if bersih.size:
            print("Jumlah       :", f"{bersih.size:,}")
            print("Minimum      :", float(np.min(bersih)))
            print("Maksimum     :", float(np.max(bersih)))
            print("Persentil 2  :", float(np.percentile(bersih, 2)))
            print("Median       :", float(np.median(bersih)))
            print("Persentil 98 :", float(np.percentile(bersih, 98)))
        else:
            print("Tidak ada nilai setelah penyaringan.")

        print("Contoh nilai :", values[:10].tolist())


def main():
    print("=" * 60)
    print("PEMERIKSAAN SKALA SELURUH BAND SENTINEL-2")
    print("=" * 60)

    for band in BANDS:
        periksa_band(band)

    print("\n" + "=" * 60)
    print("PEMERIKSAAN SELESAI")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nERROR:", type(error).__name__, "-", error)
        raise