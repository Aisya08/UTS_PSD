from __future__ import annotations

import json
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from folium.raster_layers import ImageOverlay
from rasterio.crs import CRS
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
from streamlit_folium import st_folium


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Peta Tutupan Lahan Jatim",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
      html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
      .main-title {
        font-size: 26px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 4px;
      }
      .sub-title {
        font-size: 14px;
        color: #475569;
        margin-bottom: 16px;
      }
      .badge-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 16px;
      }
      .badge-item {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        color: #1e293b;
      }
      .badge-item b {
        color: #0284c7;
      }
      .legend-container {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        background: #ffffff;
        padding: 12px 16px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin-bottom: 14px;
      }
      .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        font-weight: 600;
        color: #334155;
      }
      .swatch {
        width: 16px;
        height: 16px;
        border-radius: 4px;
        display: inline-block;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DEFINISI KELAS & WARNA
# ============================================================

CLASS_NAMES = {
    1: "Building",
    2: "Farmland",
    3: "Water",
    4: "Forest",
    5: "Mangrove",
}

CLASS_LABELS_ID = {
    1: "Perkotaan / Bangunan",
    2: "Sawah / Pertanian",
    3: "Badan Air / Laut",
    4: "Hutan Daratan",
    5: "Hutan Mangrove",
}

CLASS_COLORS = {
    1: "#E53935",  # Merah (Building)
    2: "#FDD835",  # Kuning (Farmland)
    3: "#168BFF",  # Biru (Water)
    4: "#43A047",  # Hijau (Forest)
    5: "#8E24AA",  # Ungu (Mangrove)
}

PREFERRED_FILES = [
    "hasil_klasifikasi_jatim_FINAL.tif",
    "hasil_klasifikasi_jatim_mangrove_koridor_pesisir.tif",
    "hasil_klasifikasi_jatim_mangrove_diperbaiki.tif",
    "hasil_klasifikasi_jatim_mangrove_mndwi.tif",
    "hasil_klasifikasi_jatim.tif",
]


# ============================================================
# UTILITIES PEMBACAAN RASTER
# ============================================================

def get_local_rasters() -> list[Path]:
    search_dirs = [Path("."), Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent]
    found = {}
    for d in search_dirs:
        for p in d.glob("hasil_klasifikasi_jatim*.tif"):
            found[p.resolve()] = p
        for p in d.glob("hasil_klasifikasi_jatim*.tiff"):
            found[p.resolve()] = p
    all_files = list(found.values())
    order = {name: index for index, name in enumerate(PREFERRED_FILES)}
    all_files.sort(key=lambda p: (order.get(p.name, 999), p.name.lower()))
    return all_files


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


def make_rgba(class_array: np.ndarray, selected_code: int | None, alpha: int = 205):
    rgba = np.zeros((*class_array.shape, 4), dtype=np.uint8)
    for code, hex_color in CLASS_COLORS.items():
        if selected_code is not None and code != selected_code:
            continue
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
        mask = class_array == code
        rgba[mask, 0] = rgb[0]
        rgba[mask, 1] = rgb[1]
        rgba[mask, 2] = rgb[2]
        rgba[mask, 3] = alpha
    return rgba


def build_map(
    class_array: np.ndarray,
    bounds: tuple[float, float, float, float],
    selected_code: int | None,
    opacity: float,
    show_geojson: bool,
    geojson_obj: dict | None,
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

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics",
        name="🛰️ Citra Satelit Esri",
        overlay=False,
        control=True,
        show=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="🗺️ Peta Jalan OSM",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    rgba = make_rgba(class_array, selected_code, alpha=210)
    layer_name = "Hasil Klasifikasi (Semua Kelas)" if selected_code is None else f"Hasil Klasifikasi ({CLASS_LABELS_ID.get(selected_code)})"

    ImageOverlay(
        image=rgba,
        bounds=[[south, west], [north, east]],
        origin="upper",
        opacity=opacity,
        name=layer_name,
        overlay=True,
        control=True,
        interactive=False,
        pixelated=True,
        cross_origin=False,
    ).add_to(m)

    if show_geojson and geojson_obj:
        def style_feature(feature):
            props = feature.get("properties") or {}
            label = str(props.get("class", props.get("Class", ""))).strip().lower()
            code = None
            for candidate_code, candidate_name in CLASS_NAMES.items():
                if label == candidate_name.lower():
                    code = candidate_code
                    break
            color = CLASS_COLORS.get(code, "#FFFFFF")
            return {
                "color": color,
                "weight": 2.5,
                "opacity": 0.95,
                "fillColor": color,
                "fillOpacity": 0.12,
            }

        folium.GeoJson(
            geojson_obj,
            name="📍 Poligon Training GeoJSON (ROI)",
            style_function=style_feature,
            tooltip=folium.GeoJsonTooltip(fields=["class"], aliases=["Kelas ROI:"]) if all("class" in (f.get("properties") or {}) for f in geojson_obj.get("features", [])) else None,
        ).add_to(m)

    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    m.fit_bounds([[south, west], [north, east]])
    return m


# ============================================================
# LOAD FILE RASTER & GEOJSON
# ============================================================

local_rasters = get_local_rasters()
local_geojson_candidates = [
    Path("training_jatim_mangrove_diperkecil_clean.geojson"),
    Path(__file__).resolve().parent / "training_jatim_mangrove_diperkecil_clean.geojson",
    Path(__file__).resolve().parent.parent / "training_jatim_mangrove_diperkecil_clean.geojson",
]
local_geojson = next((p for p in local_geojson_candidates if p.exists()), None)


# ============================================================
# SIDEBAR KONTROL
# ============================================================

st.sidebar.markdown("### ⚙️ Kontrol Peta")

raster_bytes = None
raster_label = None

if local_rasters:
    selected_path = local_rasters[0]
    raster_bytes = selected_path.read_bytes()
    raster_label = selected_path.name
    st.sidebar.success(f"File aktif: **{raster_label}**")
else:
    uploaded_raster = st.sidebar.file_uploader("Unggah GeoTIFF", type=["tif", "tiff"])
    if uploaded_raster:
        raster_bytes = uploaded_raster.getvalue()
        raster_label = uploaded_raster.name

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🎯 Filter Kelas Tutupan Lahan")

filter_options = {
    "Tampilkan Semua Kelas": None,
    "1. Perkotaan / Bangunan (Merah)": 1,
    "2. Sawah / Pertanian (Kuning)": 2,
    "3. Badan Air / Laut (Biru)": 3,
    "4. Hutan Daratan (Hijau)": 4,
    "5. Hutan Mangrove (Ungu)": 5,
}

selected_filter_label = st.sidebar.selectbox(
    "Pilih kelas untuk difilter:",
    options=list(filter_options.keys()),
    index=0,
)
selected_code = filter_options[selected_filter_label]

opacity = st.sidebar.slider(
    "Transparansi Layer (%)",
    min_value=0.1,
    max_value=1.0,
    value=0.85,
    step=0.05,
)

show_geojson = st.sidebar.checkbox("Tampilkan Poligon Training (ROI)", value=False)
geojson_obj = None
if show_geojson and local_geojson:
    try:
        geojson_obj = json.loads(local_geojson.read_text(encoding="utf-8"))
    except Exception:
        pass

st.sidebar.markdown("---")
if raster_bytes:
    st.sidebar.download_button(
        "⬇️ Unduh Berkas GeoTIFF",
        data=raster_bytes,
        file_name=raster_label or "hasil_klasifikasi.tif",
        mime="image/tiff",
        use_container_width=True,
    )


# ============================================================
# TAMPILAN UTAMA (FOKUS PADA PETA)
# ============================================================

st.markdown('<div class="main-title">🗺️ Peta Klasifikasi Tutupan Lahan Jawa Timur</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hasil Prediksi Citra Satelit Sentinel-2 L2A Berbasis Random Forest Classifier (Akurasi Uji: 92,76%)</div>', unsafe_allow_html=True)

# Badge info metrik
st.markdown(
    """
    <div class="badge-bar">
      <div class="badge-item">🎯 Akurasi Uji: <b>92,76%</b></div>
      <div class="badge-item">📊 Total Piksel: <b>1.186.250 Piksel</b></div>
      <div class="badge-item">📐 Resolusi: <b>~47,5 Meter / Piksel</b></div>
      <div class="badge-item">📍 Lokasi: <b>Banyuwangi & Sekitarnya</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Legenda Horizontal di atas peta
st.markdown(
    f"""
    <div class="legend-container">
      <div class="legend-item"><span class="swatch" style="background:{CLASS_COLORS[1]}"></span> Bangunan / Perkotaan</div>
      <div class="legend-item"><span class="swatch" style="background:{CLASS_COLORS[2]}"></span> Sawah / Pertanian</div>
      <div class="legend-item"><span class="swatch" style="background:{CLASS_COLORS[3]}"></span> Air / Laut</div>
      <div class="legend-item"><span class="swatch" style="background:{CLASS_COLORS[4]}"></span> Hutan Daratan</div>
      <div class="legend-item"><span class="swatch" style="background:{CLASS_COLORS[5]}"></span> Hutan Mangrove</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# TAMPILKAN PETA INTERAKTIF
if raster_bytes:
    try:
        class_array, raster_bounds = read_raster_bytes(raster_bytes)
        m = build_map(
            class_array=class_array,
            bounds=raster_bounds,
            selected_code=selected_code,
            opacity=opacity,
            show_geojson=show_geojson,
            geojson_obj=geojson_obj,
        )
        st_folium(
            m,
            height=660,
            use_container_width=True,
            returned_objects=[],
            key="map_full",
        )
    except Exception as e:
        st.error(f"Gagal memuat peta: {e}")
else:
    st.warning("Raster klasifikasi belum ditemukan.")
