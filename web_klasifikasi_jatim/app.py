from __future__ import annotations

import json
from pathlib import Path

import folium
import numpy as np
import rasterio
import streamlit as st
from folium.raster_layers import ImageOverlay
from rasterio.crs import CRS
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
from streamlit_folium import st_folium


# ============================================================
# KONFIGURASI HALAMAN (SIDEBAR DIHAPUS, FOKUS PADA PETA)
# ============================================================

st.set_page_config(
    page_title="Peta Klasifikasi Tutupan Lahan Jawa Timur",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
      html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
      
      /* Sembunyikan sidebar kiri sepenuhnya */
      [data-testid="stSidebar"], section[data-testid="stSidebar"] {
        display: none !important;
      }
      
      /* Hilangkan padding berlebih */
      .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 98% !important;
      }

      .header-box {
        background: #1e293b;
        color: #f8fafc;
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 14px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.18);
        border: 1px solid #334155;
      }
      .header-title {
        font-size: 24px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 4px;
      }
      .header-subtitle {
        font-size: 13.5px;
        color: #94a3b8;
        margin-bottom: 10px;
      }
      .instruction-pill {
        display: inline-block;
        background: #0284c7;
        color: #ffffff;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 700;
        margin-top: 4px;
      }
      
      /* Styling kontrol leaflet agar tulisan terbaca jelas */
      .leaflet-control-layers {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #0f172a !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
        padding: 10px 14px !important;
        border: 1px solid #cbd5e1 !important;
      }
      .leaflet-control-layers label {
        margin-bottom: 6px !important;
        cursor: pointer !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DEFINISI KELAS & WARNA
# ============================================================

CLASSES = [
    {
        "code": 1,
        "name_en": "Building",
        "name_id": "Bangunan / Perkotaan",
        "color": "#E53935",
        "icon": "🟥",
    },
    {
        "code": 2,
        "name_en": "Farmland",
        "name_id": "Sawah / Pertanian",
        "color": "#FDD835",
        "icon": "🟨",
    },
    {
        "code": 3,
        "name_en": "Water",
        "name_id": "Badan Air / Laut",
        "color": "#168BFF",
        "icon": "🟦",
    },
    {
        "code": 4,
        "name_en": "Forest",
        "name_id": "Hutan Daratan",
        "color": "#43A047",
        "icon": "🟩",
    },
    {
        "code": 5,
        "name_en": "Mangrove",
        "name_id": "Hutan Mangrove",
        "color": "#8E24AA",
        "icon": "🟪",
    },
]

CLASS_BY_CODE = {c["code"]: c for c in CLASSES}


# ============================================================
# UTILITIES PEMBACAAN RASTER
# ============================================================

def get_local_raster() -> Path | None:
    search_dirs = [Path("."), Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent]
    candidates = [
        "hasil_klasifikasi_jatim_FINAL.tif",
        "hasil_klasifikasi_jatim_mangrove_koridor_pesisir.tif",
        "hasil_klasifikasi_jatim_mangrove_diperbaiki.tif",
        "hasil_klasifikasi_jatim.tif",
    ]
    for name in candidates:
        for d in search_dirs:
            p = d / name
            if p.exists():
                return p.resolve()
    # Fallback to any .tif
    for d in search_dirs:
        for p in d.glob("hasil_klasifikasi_jatim*.tif"):
            return p.resolve()
    return None


def read_raster_bytes(file_bytes: bytes):
    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            data = src.read(1)
            source_crs = src.crs
            source_transform = src.transform
            source_bounds = src.bounds
            source_nodata = src.nodata

            destination_crs = CRS.from_epsg(4326)
            if source_crs == destination_crs:
                result = data
                west, south, east, north = (
                    source_bounds.left,
                    source_bounds.bottom,
                    source_bounds.right,
                    source_bounds.top,
                )
            else:
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    source_crs,
                    destination_crs,
                    src.width,
                    src.height,
                    *source_bounds,
                )
                dst = np.zeros((dst_height, dst_width), dtype=np.uint8)
                reproject(
                    source=data,
                    destination=dst,
                    src_transform=source_transform,
                    src_crs=source_crs,
                    src_nodata=source_nodata,
                    dst_transform=dst_transform,
                    dst_crs=destination_crs,
                    dst_nodata=0,
                    resampling=Resampling.nearest,
                )
                result = dst
                west, south, east, north = array_bounds(dst_height, dst_width, dst_transform)

    return result, (west, south, east, north)


def make_single_class_rgba(class_array: np.ndarray, code: int, alpha: int = 215):
    """Buat layer RGBA transparan khusus untuk satu kelas tertentu."""
    rgba = np.zeros((*class_array.shape, 4), dtype=np.uint8)
    info = CLASS_BY_CODE[code]
    hex_color = info["color"]
    rgb = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    mask = class_array == code
    rgba[mask, 0] = rgb[0]
    rgba[mask, 1] = rgb[1]
    rgba[mask, 2] = rgb[2]
    rgba[mask, 3] = alpha
    return rgba


def build_leaflet_map(
    class_array: np.ndarray,
    bounds: tuple[float, float, float, float],
    geojson_obj: dict | None,
):
    west, south, east, north = bounds
    center = [(south + north) / 2, (west + east) / 2]

    # Inisialisasi Peta Leaflet
    m = folium.Map(
        location=center,
        zoom_start=10,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    # Basemap 1: Citra Satelit Esri (Aktif secara default)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — Sentinel-2 L2A BOA",
        name="🛰️ Citra Satelit Esri",
        overlay=False,
        control=True,
        show=True,
    ).add_to(m)

    # Basemap 2: Peta Jalan OpenStreetMap
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="🗺️ Peta Jalan OSM",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    # Tambahkan masing-masing dari 5 kelas sebagai LAYER CHECKLIST INDEPENDEN
    for item in CLASSES:
        code = item["code"]
        name_id = item["name_id"]
        icon = item["icon"]
        rgba = make_single_class_rgba(class_array, code=code, alpha=215)
        layer_label = f"{icon} {name_id}"

        ImageOverlay(
            image=rgba,
            bounds=[[south, west], [north, east]],
            origin="upper",
            opacity=0.88,
            name=layer_label,
            overlay=True,
            control=True,
            show=True,  # Semua kelas dicentang secara default, bisa di-uncheck sesuka hati
            interactive=False,
            pixelated=True,
        ).add_to(m)

    # Tambahkan Layer Poligon Training GeoJSON (Default tidak dicentang)
    if geojson_obj:
        def style_feature(feature):
            props = feature.get("properties") or {}
            label = str(props.get("class", props.get("Class", ""))).strip().lower()
            color = "#FFFFFF"
            for item in CLASSES:
                if label == item["name_en"].lower():
                    color = item["color"]
                    break
            return {
                "color": color,
                "weight": 2.5,
                "opacity": 1.0,
                "fillColor": color,
                "fillOpacity": 0.15,
            }

        folium.GeoJson(
            geojson_obj,
            name="📍 Poligon Training ROI (Digitasi)",
            style_function=style_feature,
            show=False,
            tooltip=folium.GeoJsonTooltip(fields=["class"], aliases=["Kelas:"]) if any("class" in (f.get("properties") or {}) for f in geojson_obj.get("features", [])) else None,
        ).add_to(m)

    # Layer Control di pojok kanan atas (Terbuka permanen agar checklist langsung terlihat)
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    m.fit_bounds([[south, west], [north, east]])
    return m


# ============================================================
# PEMUATAN DATA
# ============================================================

raster_path = get_local_raster()

local_geojson_candidates = [
    Path("training_jatim_mangrove_diperkecil_clean.geojson"),
    Path(__file__).resolve().parent / "training_jatim_mangrove_diperkecil_clean.geojson",
    Path(__file__).resolve().parent.parent / "training_jatim_mangrove_diperkecil_clean.geojson",
]
local_geojson = next((p for p in local_geojson_candidates if p.exists()), None)
geojson_obj = None
if local_geojson:
    try:
        geojson_obj = json.loads(local_geojson.read_text(encoding="utf-8"))
    except Exception:
        pass


# ============================================================
# TAMPILAN ANTARMUKA UTAMA
# ============================================================

st.markdown(
    """
    <div class="header-box">
      <div class="header-title">🗺️ Peta Klasifikasi Tutupan Lahan Jawa Timur</div>
      <div class="header-subtitle">
        Citra Satelit Sentinel-2 L2A & Algoritma Random Forest Classifier · Akurasi Uji: 92,76% · Wilayah Banyuwangi & Sekitarnya
      </div>
      <div class="instruction-pill">
        👉 Gunakan kotak checklist di pojok kanan atas peta untuk mencentang / mematikan kelas (Sawah, Bangunan, Hutan, Mangrove, Air)
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if raster_path and raster_path.exists():
    raster_bytes = raster_path.read_bytes()
    class_array, raster_bounds = read_raster_bytes(raster_bytes)

    # Bangun dan render peta Leaflet interaktif
    folium_map = build_leaflet_map(
        class_array=class_array,
        bounds=raster_bounds,
        geojson_obj=geojson_obj,
    )

    st_folium(
        folium_map,
        height=720,
        use_container_width=True,
        returned_objects=[],
        key="leaflet_classification_map",
    )
else:
    st.error("File raster hasil_klasifikasi_jatim_FINAL.tif tidak ditemukan di repositori.")
