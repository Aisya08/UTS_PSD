import json
import geopandas as gpd
from shapely.geometry import shape


# ============================================================
# FILE
# ============================================================

input_file = "training_jatim_mangrove_diperkecil.geojson"
output_file = "training_jatim_mangrove_diperkecil_clean.geojson"


# ============================================================
# CLASS YANG DIPERBOLEHKAN
# ============================================================

kelas_valid = {
    "Building",
    "Farmland",
    "Water",
    "Forest",
    "Mangrove"
}


# ============================================================
# BACA GEOJSON LANGSUNG
# ============================================================

print("Membaca GeoJSON secara langsung...")

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)


features = data.get("features", [])

print("Total feature awal:", len(features))


# ============================================================
# PROSES SATU PER SATU
# ============================================================

features_bersih = []

dibuang = []


for i, feature in enumerate(features, start=1):

    properties = feature.get("properties") or {}

    kelas = properties.get("class")

    geometry_data = feature.get("geometry")


    # --------------------------------------------------------
    # CEK CLASS
    # --------------------------------------------------------

    if kelas not in kelas_valid:

        dibuang.append(
            f"Feature {i}: class tidak valid ({kelas})"
        )

        continue


    # --------------------------------------------------------
    # CEK GEOMETRY
    # --------------------------------------------------------

    if geometry_data is None:

        dibuang.append(
            f"Feature {i}: geometry kosong"
        )

        continue


    # --------------------------------------------------------
    # COBA BACA GEOMETRY
    # --------------------------------------------------------

    try:

        geometry = shape(geometry_data)

    except Exception as e:

        dibuang.append(
            f"Feature {i}: geometry rusak ({e})"
        )

        continue


    # --------------------------------------------------------
    # CEK GEOMETRY KOSONG
    # --------------------------------------------------------

    if geometry.is_empty:

        dibuang.append(
            f"Feature {i}: geometry kosong"
        )

        continue


    # --------------------------------------------------------
    # CEK GEOMETRY VALID
    # --------------------------------------------------------

    if not geometry.is_valid:

        dibuang.append(
            f"Feature {i}: geometry tidak valid"
        )

        continue


    # --------------------------------------------------------
    # HANYA TERIMA POLYGON
    # --------------------------------------------------------

    if geometry.geom_type not in [
        "Polygon",
        "MultiPolygon"
    ]:

        dibuang.append(
            f"Feature {i}: bukan Polygon ({geometry.geom_type})"
        )

        continue


    # --------------------------------------------------------
    # MASUKKAN KE DATA BERSIH
    # --------------------------------------------------------

    features_bersih.append({

        "type": "Feature",

        "properties": {
            "class": kelas
        },

        "geometry": geometry.__geo_interface__

    })


# ============================================================
# BUAT GEOJSON BARU
# ============================================================

geojson_bersih = {

    "type": "FeatureCollection",

    "features": features_bersih

}


# ============================================================
# SIMPAN
# ============================================================

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        geojson_bersih,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# HASIL
# ============================================================

print("\n========================================")
print("PEMBERSIHAN SELESAI")
print("========================================")

print(
    "Total feature awal :",
    len(features)
)

print(
    "Total feature valid:",
    len(features_bersih)
)

print(
    "Total dibuang      :",
    len(dibuang)
)


# ============================================================
# JUMLAH PER KELAS
# ============================================================

print("\nJumlah polygon per kelas:")

jumlah_kelas = {}

for feature in features_bersih:

    kelas = feature["properties"]["class"]

    jumlah_kelas[kelas] = jumlah_kelas.get(kelas, 0) + 1


for kelas in [
    "Building",
    "Farmland",
    "Water",
    "Forest",
    "Mangrove"
]:

    print(
        f"{kelas:10s}:",
        jumlah_kelas.get(kelas, 0)
    )


# ============================================================
# FEATURE YANG DIBUANG
# ============================================================

print("\nFeature yang dibuang:")

if len(dibuang) == 0:

    print("Tidak ada.")

else:

    for item in dibuang:

        print("-", item)


# ============================================================
# OUTPUT
# ============================================================

print("\nFile output:")

print(output_file)

print("\n========================================")
print("SELESAI")
print("========================================")