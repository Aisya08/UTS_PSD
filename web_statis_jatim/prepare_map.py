import os
import sys
import json
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import rasterio
# pyrefly: ignore [missing-import]
from PIL import Image

# ============================================================
# KONFIGURASI DAN PRIORITAS RASTER
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

RASTER_PRIORITIES = [
    "hasil_klasifikasi_jatim_FINAL.tif",
    "hasil_klasifikasi_jatim_mangrove_koridor_pesisir.tif",
    "hasil_klasifikasi_jatim.tif"
]

ASSETS_DIR = os.path.join(SCRIPT_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# Definisi Kelas Tutupan Lahan (Kode 1-5)
# Sesuai spesifikasi warna & emoji:
# 1: Building (Merah)
# 2: Farmland (Kuning)
# 3: Water (Biru)
# 4: Forest (Hijau)
# 5: Mangrove (Ungu)
CLASSES = {
    1: {
        "id": "building",
        "name": "Perkotaan / Lahan Terbangun",
        "en": "Building",
        "rgb": (229, 57, 53),       # #E53935
        "hex": "#E53935",
        "file": "building.png"
    },
    2: {
        "id": "farmland",
        "name": "Sawah / Lahan Pertanian",
        "en": "Farmland",
        "rgb": (251, 192, 45),     # #FBC02D
        "hex": "#FBC02D",
        "file": "farmland.png"
    },
    3: {
        "id": "water",
        "name": "Perairan",
        "en": "Water",
        "rgb": (30, 136, 229),      # #1E88E5
        "hex": "#1E88E5",
        "file": "water.png"
    },
    4: {
        "id": "forest",
        "name": "Hutan",
        "en": "Forest",
        "rgb": (46, 125, 50),       # #2E7D32
        "hex": "#2E7D32",
        "file": "forest.png"
    },
    5: {
        "id": "mangrove",
        "name": "Hutan Mangrove",
        "en": "Mangrove",
        "rgb": (142, 36, 170),      # #8E24AA (Ungu)
        "hex": "#8E24AA",
        "file": "mangrove.png"
    }
}


def find_input_raster():
    """Mencari file raster klasifikasi berdasarkan urutan prioritas."""
    search_dirs = [SCRIPT_DIR, PROJECT_DIR]
    
    for filename in RASTER_PRIORITIES:
        for folder in search_dirs:
            candidate = os.path.join(folder, filename)
            if os.path.exists(candidate):
                return candidate, filename
                
    raise FileNotFoundError(
        "Tidak ada file raster klasifikasi yang ditemukan dari daftar prioritas: "
        + ", ".join(RASTER_PRIORITIES)
    )


def prepare_map_assets():
    print("=" * 65)
    print("PREPARE MAP - KONVERSI RASTER KE OVERLAY PNG STATIS")
    print("=" * 65)

    raster_path, raster_filename = find_input_raster()
    print(f"\n[1] Menggunakan raster terpilih:")
    print(f"    Path: {raster_path}")
    print(f"    File: {raster_filename}")

    with rasterio.open(raster_path) as src:
        data = src.read(1)
        bounds = src.bounds
        crs = src.crs
        width = src.width
        height = src.height

        left = float(bounds.left)
        bottom = float(bounds.bottom)
        right = float(bounds.right)
        top = float(bounds.top)

        print(f"\n[2] Metadata Raster:")
        print(f"    Dimensi : {width} x {height} px")
        print(f"    CRS     : {crs}")
        print(f"    Batas Barat (West)  : {left:.7f}")
        print(f"    Batas Selatan (South): {bottom:.7f}")
        print(f"    Batas Timur (East)  : {right:.7f}")
        print(f"    Batas Utara (North) : {top:.7f}")

    # Bounds dalam format Leaflet: [[South, West], [North, East]]
    leaflet_bounds = [
        [bottom, left],
        [top, right]
    ]
    center_lat = (bottom + top) / 2.0
    center_lon = (left + right) / 2.0

    print("\n[3] Mengekstrak overlay PNG transparan per kelas...")
    class_stats = {}

    for code, info in CLASSES.items():
        mask = (data == code)
        pixel_count = int(np.sum(mask))
        percentage = (pixel_count / (width * height)) * 100.0
        
        class_stats[info["id"]] = {
            "code": code,
            "name": info["name"],
            "en": info["en"],
            "color": info["hex"],
            "file": info["file"],
            "pixels": pixel_count,
            "percentage": round(percentage, 2)
        }

        print(f"    - Kelas {code} ({info['en']}): {pixel_count:,} piksel ({percentage:.2f}%)")

        # Buat citra RGBA transparan
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Atur warna RGB untuk piksel kelas terkait
        rgba[mask, 0] = info["rgb"][0]
        rgba[mask, 1] = info["rgb"][1]
        rgba[mask, 2] = info["rgb"][2]
        
        # Alpha 255 (solid di PNG, transparansi overlay diatur fleksibel di Leaflet)
        rgba[mask, 3] = 255

        # Simpan ke PNG
        out_path = os.path.join(ASSETS_DIR, info["file"])
        img = Image.fromarray(rgba, mode="RGBA")
        img.save(out_path, optimize=True)
        print(f"      Tersimpan: assets/{info['file']}")

    # Simpan metadata bounds dan kelas ke bounds.json
    metadata = {
        "source_file": raster_filename,
        "crs": str(crs),
        "dimensions": {
            "width": width,
            "height": height
        },
        "bounds": leaflet_bounds,
        "center": [center_lat, center_lon],
        "classes": class_stats
    }

    metadata_path = os.path.join(ASSETS_DIR, "bounds.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[4] Metadata bounds tersimpan di: assets/bounds.json")
    print("=" * 65)
    print("PROSES SELESAI DENGAN SUKSES!")
    print("=" * 65)


if __name__ == "__main__":
    prepare_map_assets()
