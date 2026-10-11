import os
import base64
import numpy as np
import rasterio
from PIL import Image


# ============================================================
# 1. PENGATURAN
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_TIF = os.path.join(
    BASE_DIR,
    "hasil_klasifikasi_jatim_smooth.tif"
)

OUTPUT_HTML = os.path.join(
    BASE_DIR,
    "index_satelit.html"
)

OVERLAY_DIR = os.path.join(
    BASE_DIR,
    "leaflet_layers"
)

os.makedirs(OVERLAY_DIR, exist_ok=True)


# ============================================================
# 2. KELAS DAN WARNA
# ============================================================

CLASSES = {
    1: ("Building", (255, 0, 0)),
    2: ("Farmland", (255, 255, 0)),
    3: ("Water", (0, 0, 255)),
    4: ("Forest", (0, 128, 0)),
    5: ("Mangrove", (0, 100, 0)),
}


# ============================================================
# 3. BACA RASTER
# ============================================================

print("=" * 60)
print("MEMBUAT LEAFLET - SATELLITE + KLASIFIKASI")
print("=" * 60)

with rasterio.open(INPUT_TIF) as src:

    data = src.read(1)

    bounds = src.bounds

    left = bounds.left
    bottom = bounds.bottom
    right = bounds.right
    top = bounds.top

    width = src.width
    height = src.height

    print("\nRaster:")
    print(f"Ukuran : {width} x {height}")
    print(f"CRS    : {src.crs}")

    print("\nBounds:")
    print(f"West  : {left}")
    print(f"South : {bottom}")
    print(f"East  : {right}")
    print(f"North : {top}")


# ============================================================
# 4. BUAT PNG SETIAP KELAS
# ============================================================

print("\nMembuat overlay kelas...")

layer_data = {}

for code, (class_name, color) in CLASSES.items():

    print(f"  {code} - {class_name}")

    rgba = np.zeros(
        (height, width, 4),
        dtype=np.uint8
    )

    mask = data == code

    rgba[mask, 0] = color[0]
    rgba[mask, 1] = color[1]
    rgba[mask, 2] = color[2]

    # Transparansi
    rgba[mask, 3] = 120

    image = Image.fromarray(
        rgba,
        mode="RGBA"
    )

    filename = (
        class_name.lower()
        + ".png"
    )

    path = os.path.join(
        OVERLAY_DIR,
        filename
    )

    image.save(path)

    layer_data[class_name] = path


# ============================================================
# 5. BASE64
# ============================================================

print("\nMengubah layer menjadi Base64...")

base64_layers = {}

for class_name, path in layer_data.items():

    with open(path, "rb") as f:

        encoded = base64.b64encode(
            f.read()
        ).decode("utf-8")

    base64_layers[class_name] = encoded


# ============================================================
# 6. JAVASCRIPT OVERLAY
# ============================================================

layer_js = ""

for code, (class_name, color) in CLASSES.items():

    encoded = base64_layers[class_name]

    layer_js += f"""
var layer_{code} = L.imageOverlay(
    "data:image/png;base64,{encoded}",
    bounds,
    {{
        opacity: 0.75,
        interactive: false
    }}
);
"""


# ============================================================
# 7. HTML
# ============================================================

html = f"""<!DOCTYPE html>

<html lang="id">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
Klasifikasi Tutupan Lahan Jawa Timur
</title>

<link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
/>

<style>

html,
body {{
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
}}

#map {{
    width: 100%;
    height: 100%;
}}

.title {{
    position: absolute;
    top: 15px;
    left: 55px;
    z-index: 1000;

    background: rgba(255,255,255,0.95);

    padding: 12px 18px;

    border-radius: 10px;

    box-shadow:
        0 2px 10px rgba(0,0,0,0.25);

    font-family: Arial, sans-serif;
}}

.title h2 {{
    margin: 0 0 5px 0;
    font-size: 20px;
}}

.title p {{
    margin: 0;
    font-size: 13px;
}}

.legend {{
    background: white;
    padding: 12px;

    border-radius: 8px;

    box-shadow:
        0 2px 8px rgba(0,0,0,0.25);

    font-family: Arial, sans-serif;
}}

.legend-title {{
    font-weight: bold;
    margin-bottom: 7px;
}}

.legend-item {{
    margin: 4px 0;
}}

.legend-color {{
    display: inline-block;

    width: 15px;
    height: 15px;

    margin-right: 6px;

    vertical-align: middle;
}}

</style>

</head>


<body>


<div class="title">

<h2>
Klasifikasi Tutupan Lahan
</h2>

<p>
Jawa Timur • Sentinel-2 • Random Forest
</p>

</div>


<div id="map"></div>


<script
src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js">
</script>


<script>


// ============================================================
// BOUNDS RASTER
// ============================================================

var bounds = [
    [{bottom}, {left}],
    [{top}, {right}]
];


// ============================================================
// MAP
// ============================================================

var map = L.map("map");


// ============================================================
// SATELLITE BASEMAP
// ============================================================

var satellite = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}",
    {{
        attribution:
        "Tiles © Esri"
    }}
).addTo(map);


// Zoom ke area hasil klasifikasi

map.fitBounds(bounds);


// ============================================================
// OVERLAY KLASIFIKASI
// ============================================================

{layer_js}


// ============================================================
// KONTROL KELAS
// ============================================================

var layers = {{

    "Building": layer_1,

    "Farmland": layer_2,

    "Water": layer_3,

    "Forest": layer_4,

    "Mangrove": layer_5

}};


// ============================================================
// CUSTOM CONTROL
// HANYA SATU KELAS AKTIF
// ============================================================

var control = L.control({{
    position: "topright"
}});


control.onAdd = function(map) {{

    var div = L.DomUtil.create(
        "div",
        "leaflet-control"
    );

    div.style.background = "white";
    div.style.padding = "12px";
    div.style.borderRadius = "8px";
    div.style.boxShadow =
        "0 2px 8px rgba(0,0,0,0.3)";
    div.style.fontFamily =
        "Arial, sans-serif";

    div.innerHTML =

        "<b>Pilih Kelas</b><br><br>" +

        '<label>' +
        '<input type="radio" name="kelas" value="none" checked> ' +
        'Tidak ada' +
        '</label><br>' +

        '<label>' +
        '<input type="radio" name="kelas" value="Building"> ' +
        '🔴 Building' +
        '</label><br>' +

        '<label>' +
        '<input type="radio" name="kelas" value="Farmland"> ' +
        '🟡 Farmland' +
        '</label><br>' +

        '<label>' +
        '<input type="radio" name="kelas" value="Water"> ' +
        '🔵 Water' +
        '</label><br>' +

        '<label>' +
        '<input type="radio" name="kelas" value="Forest"> ' +
        '🟢 Forest' +
        '</label><br>' +

        '<label>' +
        '<input type="radio" name="kelas" value="Mangrove"> ' +
        '🌿 Mangrove' +
        '</label>';


    L.DomEvent.disableClickPropagation(div);

    return div;
}};


control.addTo(map);


// ============================================================
// EVENT PILIH KELAS
// ============================================================

document
.querySelectorAll('input[name="kelas"]')
.forEach(function(radio) {{

    radio.addEventListener(
        "change",
        function() {{

            // Hapus semua layer

            Object.values(layers).forEach(
                function(layer) {{

                    if (map.hasLayer(layer)) {{

                        map.removeLayer(layer);

                    }}

                }}
            );


            // Tambahkan kelas yang dipilih

            var selected = this.value;

            if (
                selected !== "none"
                &&
                layers[selected]
            ) {{

                layers[selected].addTo(map);

            }}

        }}
    );

}});


// ============================================================
// LEGEND
// ============================================================

var legend = L.control({{
    position: "bottomright"
}});


legend.onAdd = function(map) {{

    var div = L.DomUtil.create(
        "div",
        "legend"
    );

    div.innerHTML =

        '<div class="legend-title">' +
        'Kelas Tutupan Lahan' +
        '</div>' +

        '<div class="legend-item">' +
        '<span class="legend-color" ' +
        'style="background:red"></span>' +
        'Building' +
        '</div>' +

        '<div class="legend-item">' +
        '<span class="legend-color" ' +
        'style="background:yellow"></span>' +
        'Farmland' +
        '</div>' +

        '<div class="legend-item">' +
        '<span class="legend-color" ' +
        'style="background:blue"></span>' +
        'Water' +
        '</div>' +

        '<div class="legend-item">' +
        '<span class="legend-color" ' +
        'style="background:green"></span>' +
        'Forest' +
        '</div>' +

        '<div class="legend-item">' +
        '<span class="legend-color" ' +
        'style="background:#006400"></span>' +
        'Mangrove' +
        '</div>';

    return div;

}};


legend.addTo(map);


</script>

</body>

</html>
"""


# ============================================================
# 8. SIMPAN
# ============================================================

with open(
    OUTPUT_HTML,
    "w",
    encoding="utf-8"
) as f:

    f.write(html)


print("\n" + "=" * 60)
print("LEAFLET SATELIT BERHASIL DIBUAT")
print("=" * 60)

print("\nFile:")
print(OUTPUT_HTML)

print("\nFitur:")
print("- Citra satelit")
print("- Building")
print("- Farmland")
print("- Water")
print("- Forest")
print("- Mangrove")
print("- Hanya satu kelas aktif")
print("=" * 60)