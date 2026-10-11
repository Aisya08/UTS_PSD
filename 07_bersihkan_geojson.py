import json
from shapely.geometry import shape

INPUT = "training_jatim_building_tambahan.geojson"
OUTPUT = "training_jatim_building_clean.geojson"

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

valid_features = []
removed = []

allowed_classes = {
    "Building",
    "Farmland",
    "Water",
    "Forest",
    "Mangrove"
}

for i, feature in enumerate(data["features"], start=1):

    props = feature.get("properties", {})
    kelas = props.get("class")
    geometry = feature.get("geometry")

    # Cek class
    if kelas not in allowed_classes:
        removed.append((i, kelas, "class kosong/tidak dikenal"))
        continue

    # Cek geometry
    if geometry is None:
        removed.append((i, kelas, "geometry kosong"))
        continue

    try:
        geom = shape(geometry)

        if not geom.is_valid:
            removed.append((i, kelas, "geometry tidak valid"))
            continue

        if geom.is_empty:
            removed.append((i, kelas, "geometry kosong"))
            continue

        valid_features.append(feature)

    except Exception as e:
        removed.append((i, kelas, "geometry rusak"))

# Buat GeoJSON baru
data["features"] = valid_features

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(
        data,
        f,
        ensure_ascii=False,
        indent=2
    )

print("=" * 60)
print("PEMBERSIHAN GEOJSON SELESAI")
print("=" * 60)

print(f"Total awal   : {len(data['features']) + len(removed)}")
print(f"Total valid  : {len(valid_features)}")
print(f"Total dibuang: {len(removed)}")

print("\nFeature yang dibuang:")

for item in removed:
    print(
        f"Feature {item[0]} | "
        f"Class: {item[1]} | "
        f"Alasan: {item[2]}"
    )

print("\nFile hasil:")
print(OUTPUT)