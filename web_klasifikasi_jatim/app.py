from __future__ import annotations

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
# KONFIGURASI HALAMAN (FULLSCREEN PETA, SIDEBAR DIHAPUS)
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
      html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
      }
      
      /* Sembunyikan sidebar kiri sepenuhnya */
      [data-testid="stSidebar"], section[data-testid="stSidebar"] {
        display: none !important;
      }
      
      /* Hilangkan margin dan padding atas agar peta langsung tampil penuh */
      .block-container {
        padding-top: 0.2rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
      }
      
      /* Sembunyikan header default streamlit yang tidak perlu */
      header[data-testid="stHeader"] {
        background: transparent !important;
        height: 1.5rem !important;
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
):
    west, south, east, north = bounds
    center = [(south + north) / 2, (west + east) / 2]

    m = folium.Map(
        location=center,
        zoom_start=10,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    # Injeksi CSS Modern langsung ke dalam Header Peta Leaflet agar kotak layer terlihat sangat rapi & estetik
    custom_leaflet_css = folium.Element("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
      
      /* Kotak Kontrol Layer Leaflet Elegan & Modern */
      .leaflet-control-layers {
        background: rgba(255, 255, 255, 0.96) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(203, 213, 225, 0.9) !important;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.22), 0 2px 10px rgba(0, 0, 0, 0.08) !important;
        padding: 14px 18px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        min-width: 260px !important;
        color: #0f172a !important;
      }
      
      /* Judul Bagian Basemap */
      .leaflet-control-layers-base::before {
        content: "BASEMAP SATELIT";
        display: block;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1px;
        color: #64748b;
        margin-bottom: 8px;
        padding-left: 4px;
      }
      
      .leaflet-control-layers-base label {
        display: flex !important;
        align-items: center !important;
        padding: 6px 10px !important;
        border-radius: 8px !important;
        font-size: 13.5px !important;
        font-weight: 600 !important;
        color: #1e293b !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        margin-bottom: 2px !important;
      }
      
      .leaflet-control-layers-base label:hover {
        background: #f1f5f9 !important;
      }
      
      /* Garis Pemisah Antara Basemap dan Kelas */
      .leaflet-control-layers-separator {
        border-top: 1.5px solid #e2e8f0 !important;
        margin: 12px 0 10px 0 !important;
      }
      
      /* Judul Bagian Checklist Kelas */
      .leaflet-control-layers-overlays::before {
        content: "TAMPILKAN KELAS";
        display: block;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1px;
        color: #64748b;
        margin-bottom: 8px;
        padding-left: 4px;
      }
      
      .leaflet-control-layers-overlays label {
        display: flex !important;
        align-items: center !important;
        padding: 6px 10px !important;
        border-radius: 8px !important;
        font-size: 13.5px !important;
        font-weight: 600 !important;
        color: #0f172a !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        margin-bottom: 3px !important;
      }
      
      .leaflet-control-layers-overlays label:hover {
        background: #f1f5f9 !important;
      }
      
      /* Desain Checklist Checkbox & Radio Button */
      .leaflet-control-layers-selector {
        width: 18px !important;
        height: 18px !important;
        margin-right: 12px !important;
        cursor: pointer !important;
        accent-color: #2563eb !important;
      }
      
      /* Tombol Zoom Modern */
      .leaflet-bar {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.2) !important;
        border: 1px solid rgba(226, 232, 240, 0.9) !important;
      }
      .leaflet-bar a {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #0f172a !important;
        font-weight: bold !important;
        width: 34px !important;
        height: 34px !important;
        line-height: 34px !important;
      }
    </style>
    """)
    m.get_root().header.add_child(custom_leaflet_css)

    # Basemap 1: Citra Satelit Esri (Aktif default)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri",
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
            show=True,
            interactive=False,
            pixelated=True,
        ).add_to(m)

    # Kotakan Batas Wilayah Kajian Banyuwangi (Bounding Box Garis Putus-Putus Kontras)
    bbox_group = folium.FeatureGroup(name="🔲 Kotak Wilayah Banyuwangi", show=True)
    folium.Rectangle(
        bounds=[[south, west], [north, east]],
        color="#f43f5e",
        weight=3.5,
        dash_array="10, 6",
        fill=False,
        tooltip="📍 Kotak Batas Wilayah Kajian: Banyuwangi & Sekitarnya (~2.683 km²)",
        popup="<b>Batas Wilayah Kajian</b><br>Kabupaten Banyuwangi dan Sekitarnya<br>Citra Sentinel-2 L2A (1.186.250 Piksel)",
    ).add_to(bbox_group)
    bbox_group.add_to(m)

    # Layer Control di pojok kanan atas (Terbuka permanen)
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    m.fit_bounds([[south, west], [north, east]])
    return m


# ============================================================
# PEMUATAN DATA & TAMPILAN PETA LANGSUNG
# ============================================================

raster_path = get_local_raster()

if raster_path and raster_path.exists():
    raster_bytes = raster_path.read_bytes()
    class_array, raster_bounds = read_raster_bytes(raster_bytes)

    # Bangun peta Leaflet interaktif
    folium_map = build_leaflet_map(
        class_array=class_array,
        bounds=raster_bounds,
    )

    # Render peta layar penuh
    st_folium(
        folium_map,
        height=780,
        use_container_width=True,
        returned_objects=[],
        key="leaflet_classification_map",
    )
else:
    st.error("File raster hasil_klasifikasi_jatim_FINAL.tif tidak ditemukan di repositori.")
