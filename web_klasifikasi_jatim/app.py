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
# PAGE CONFIG & THEME
# ============================================================

st.set_page_config(
    page_title="Jatim LandCover AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .stApp { background: #f5f8fc; }
      [data-testid="stHeader"] { background: rgba(245,248,252,.92); }
      [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e4ebf3; }
      .hero {
        background: linear-gradient(120deg, #0b2447 0%, #124b79 55%, #168aad 100%);
        border-radius: 22px; padding: 28px 30px; color: white; margin-bottom: 18px;
        box-shadow: 0 12px 34px rgba(11,36,71,.13);
      }
      .hero-kicker { font-size: 12px; text-transform: uppercase; letter-spacing: 2px; opacity: .78; font-weight: 700; }
      .hero h1 { color: white; font-family: 'Manrope', sans-serif; font-size: clamp(26px, 3vw, 38px); margin: 7px 0 8px 0; line-height: 1.12; }
      .hero p { color: #d9eaf5; margin: 0; font-size: 15px; max-width: 850px; }
      .soft-card { background: white; border: 1px solid #e4ebf3; padding: 17px 18px; border-radius: 16px; min-height: 102px; }
      .metric-label { font-size: 12px; color: #63758b; font-weight: 700; text-transform: uppercase; letter-spacing: .7px; }
      .metric-value { font-family: 'Manrope', sans-serif; font-size: 27px; color: #112c4b; font-weight: 800; margin-top: 7px; }
      .metric-note { font-size: 12px; color: #748399; margin-top: 3px; }
      .section-title { font-family: 'Manrope', sans-serif; font-size: 22px; font-weight: 800; color: #132e4e; margin: 4px 0 5px; }
      .section-subtitle { color: #64758b; font-size: 13px; margin-bottom: 14px; }
      .info-note { background: #eef6fb; border: 1px solid #d6eaf6; padding: 12px 14px; border-radius: 12px; color: #23445f; font-size: 13px; }
      .warning-note { background: #fff8e9; border: 1px solid #f1dfb5; padding: 12px 14px; border-radius: 12px; color: #72521a; font-size: 13px; }
      .stTabs [data-baseweb="tab-list"] { gap: 8px; }
      .stTabs [data-baseweb="tab"] { background: white; border-radius: 10px 10px 0 0; padding: 10px 17px; }
      div[data-testid="stMetric"] { background: white; border: 1px solid #e4ebf3; padding: 12px 15px; border-radius: 14px; }
      .small-muted { color: #708198; font-size: 12px; }
      .legend-pill { display: inline-block; margin: 3px 7px 3px 0; border-radius: 7px; padding: 5px 9px; background: white; border: 1px solid #e3e9f0; font-size: 12px; color: #263e58; }
      .swatch { display: inline-block; width: 9px; height: 9px; border-radius: 3px; margin-right: 6px; }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA DEFINITIONS
# ============================================================

CLASS_NAMES = {
    1: "Building",
    2: "Farmland",
    3: "Water",
    4: "Forest",
    5: "Mangrove",
}

CLASS_LABELS_ID = {
    1: "Bangunan",
    2: "Lahan Pertanian",
    3: "Perairan",
    4: "Hutan",
    5: "Hutan Mangrove",
}

CLASS_COLORS = {
    1: "#E53935",  # Building
    2: "#FDD835",  # Farmland
    3: "#168BFF",  # Water
    4: "#43A047",  # Forest
    5: "#8E24AA",  # Mangrove
}

CLASS_DESCRIPTIONS = {
    1: "Area terbangun seperti permukiman, bangunan, dan kawasan perkotaan.",
    2: "Lahan pertanian, sawah, ladang, dan area budidaya.",
    3: "Area perairan yang terdeteksi pada citra, seperti laut atau badan air.",
    4: "Tutupan vegetasi berhutan selain kelas mangrove.",
    5: "Vegetasi mangrove yang diprediksi model dan ditampilkan pada peta.",
}

PREFERRED_FILES = [
    "hasil_klasifikasi_jatim_FINAL.tif",
    "hasil_klasifikasi_jatim_mangrove_koridor_pesisir.tif",
    "hasil_klasifikasi_jatim_mangrove_diperbaiki.tif",
    "hasil_klasifikasi_jatim_mangrove_mndwi.tif",
    "hasil_klasifikasi_jatim.tif",
]


# ============================================================
# RASTER UTILITIES
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
    """Read a one-band land-cover raster and return EPSG:4326 array/bounds."""
    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            data = src.read(1)
            source_crs = src.crs
            source_transform = src.transform
            source_bounds = src.bounds
            source_nodata = src.nodata
            source_dtype = src.dtypes[0]

            if source_crs is None:
                raise ValueError("Raster tidak memiliki CRS/georeferensi.")

            destination_crs = CRS.from_epsg(4326)

            if source_crs == destination_crs:
                result = data.astype(np.uint8, copy=False)
                west, south, east, north = array_bounds(
                    src.height, src.width, source_transform
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
                west, south, east, north = array_bounds(
                    dst_height, dst_width, dst_transform
                )

    # A class raster should contain codes 0-5 only.
    present = set(np.unique(result).tolist())
    unknown = sorted(present - {0, 1, 2, 3, 4, 5})
    if unknown:
        raise ValueError(
            "Raster tidak terlihat seperti hasil klasifikasi kelas 0–5. "
            f"Kode yang tidak dikenal: {unknown[:12]}"
        )

    return result, (west, south, east, north), source_dtype


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
        attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Citra satelit",
        overlay=False,
        control=True,
        show=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Peta jalan",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    rgba = make_rgba(class_array, selected_code, alpha=205)
    if selected_code is None:
        layer_name = "Semua kelas tutupan lahan"
    else:
        layer_name = f"Kelas: {CLASS_LABELS_ID[selected_code]}"

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
            label = props.get("class", props.get("Class", ""))
            code = None
            for candidate_code, candidate_name in CLASS_NAMES.items():
                if str(label).strip().lower() == candidate_name.lower():
                    code = candidate_code
                    break
            color = CLASS_COLORS.get(code, "#FFFFFF")
            return {
                "color": color,
                "weight": 2,
                "opacity": 0.9,
                "fillColor": color,
                "fillOpacity": 0.08,
            }

        folium.GeoJson(
            geojson_obj,
            name="Polygon training GeoJSON",
            style_function=style_feature,
            tooltip=folium.GeoJsonTooltip(
                fields=["class"],
                aliases=["Kelas:"],
                sticky=False,
            ) if all(
                "class" in (feature.get("properties") or {})
                for feature in geojson_obj.get("features", [])
            ) else None,
        ).add_to(m)

    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    m.fit_bounds([[south, west], [north, east]])
    return m


# ============================================================
# SIDEBAR — DATA INPUT
# ============================================================

st.sidebar.markdown("## 🛰️ Jatim LandCover AI")
st.sidebar.caption("Pemetaan tutupan lahan berbasis Sentinel-2")
st.sidebar.markdown("---")

local_rasters = get_local_rasters()
local_geojson_candidates = [
    Path("training_jatim_mangrove_diperkecil_clean.geojson"),
    Path("training_jatim_final_clean.geojson"),
    Path("training_jatim_50_polygon_clean.geojson"),
]
local_geojson = next((p for p in local_geojson_candidates if p.exists()), None)

source_mode = st.sidebar.radio(
    "Sumber raster klasifikasi",
    options=["File di folder aplikasi", "Unggah GeoTIFF"],
    index=0 if local_rasters else 1,
)

raster_bytes = None
raster_label = None

if source_mode == "File di folder aplikasi":
    if local_rasters:
        preferred_names = [p.name for p in local_rasters]
        selected_local_name = st.sidebar.selectbox(
            "Pilih hasil klasifikasi",
            preferred_names,
            index=0,
            help="File FINAL diprioritaskan jika tersedia di folder aplikasi.",
        )
        selected_path = next(p for p in local_rasters if p.name == selected_local_name)
        try:
            raster_bytes = selected_path.read_bytes()
            raster_label = selected_path.name
        except OSError as exc:
            st.sidebar.error(f"File tidak bisa dibaca: {exc}")
    else:
        st.sidebar.info(
            "Belum ada file hasil klasifikasi di folder aplikasi. "
            "Pindahkan hasil_klasifikasi_jatim_FINAL.tif ke folder yang sama, "
            "atau pilih mode Unggah GeoTIFF."
        )
else:
    uploaded_raster = st.sidebar.file_uploader(
        "Unggah raster hasil klasifikasi (.tif/.tiff)",
        type=["tif", "tiff"],
        key="raster_uploader",
        help="Unggah raster dengan kode kelas 1–5 dan CRS yang tersedia.",
    )
    if uploaded_raster is not None:
        raster_bytes = uploaded_raster.getvalue()
        raster_label = uploaded_raster.name

st.sidebar.markdown("---")
show_geojson = st.sidebar.checkbox(
    "Tampilkan polygon training",
    value=False,
    help="Menampilkan GeoJSON training sebagai garis batas di atas peta.",
)

geojson_obj = None
if show_geojson:
    geojson_source_mode = st.sidebar.radio(
        "Sumber polygon",
        options=["GeoJSON lokal", "Unggah GeoJSON"],
        index=0 if local_geojson else 1,
    )
    if geojson_source_mode == "GeoJSON lokal":
        if local_geojson:
            try:
                geojson_obj = json.loads(local_geojson.read_text(encoding="utf-8"))
                st.sidebar.caption(f"GeoJSON: {local_geojson.name}")
            except (OSError, json.JSONDecodeError) as exc:
                st.sidebar.error(f"GeoJSON tidak dapat dibaca: {exc}")
        else:
            st.sidebar.info("Tidak ada GeoJSON training di folder aplikasi; pilih Unggah GeoJSON.")
    else:
        uploaded_geojson = st.sidebar.file_uploader(
            "Unggah GeoJSON training",
            type=["geojson", "json"],
            key="geojson_uploader",
        )
        if uploaded_geojson is not None:
            try:
                geojson_obj = json.loads(uploaded_geojson.getvalue().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                st.sidebar.error(f"GeoJSON tidak valid: {exc}")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Remote Sensing · Machine Learning · Leaflet</div>
      <h1>Jatim LandCover AI</h1>
      <p>Dashboard peta interaktif untuk melihat distribusi tutupan lahan di wilayah kajian Jawa Timur dari citra Sentinel-2 dan klasifikasi Random Forest.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


if raster_bytes is None:
    st.markdown(
        """
        <div class="info-note">
          <b>Mulai dengan memuat raster klasifikasi.</b><br>
          Pilih file <code>hasil_klasifikasi_jatim_FINAL.tif</code> dari folder aplikasi, """
        + """
          atau buka menu di sidebar dan unggah file GeoTIFF hasil klasifikasimu.
          Web ini menampilkan raster yang kamu miliki; data contoh tidak dibuat-buat.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ============================================================
# READ RASTER
# ============================================================

try:
    class_array, raster_bounds, source_dtype = read_raster_bytes(raster_bytes)
except Exception as exc:
    st.error(f"Raster tidak dapat diproses: {exc}")
    st.info(
        "Pastikan file yang diunggah adalah raster hasil klasifikasi satu band, "
        "dengan kode 1=Building, 2=Farmland, 3=Water, 4=Forest, 5=Mangrove."
    )
    st.stop()

west, south, east, north = raster_bounds

# Valid class pixels only. Code 0 is treated as NoData/background.
valid_class_mask = np.isin(class_array, list(CLASS_NAMES.keys()))
valid_pixels = int(np.count_nonzero(valid_class_mask))
if valid_pixels == 0:
    st.error("Raster tidak berisi kode kelas 1–5.")
    st.stop()

class_counts = {
    code: int(np.count_nonzero(class_array == code))
    for code in CLASS_NAMES
}
total_class_pixels = sum(class_counts.values())

stats_rows = []
for code, english_name in CLASS_NAMES.items():
    count = class_counts[code]
    percent = (count / total_class_pixels * 100) if total_class_pixels else 0
    stats_rows.append({
        "Kode": code,
        "Kelas": CLASS_LABELS_ID[code],
        "Nama kelas": english_name,
        "Jumlah piksel": count,
        "Persentase (%)": round(percent, 2),
        "Warna": CLASS_COLORS[code],
    })
stats_df = pd.DataFrame(stats_rows)


# ============================================================
# DASHBOARD METRICS
# ============================================================

st.markdown('<div class="section-title">Ringkasan peta</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-subtitle">Sumber aktif: <b>{raster_label}</b> · Raster {class_array.shape[1]:,} × {class_array.shape[0]:,} piksel · EPSG:4326 pada peta</div>',
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("Piksel berkelas", f"{total_class_pixels:,}")
with metric_cols[1]:
    st.metric("Jumlah kelas", "5")
with metric_cols[2]:
    st.metric("Piksel mangrove", f"{class_counts[MANGROVE]:,}", f"{(class_counts[MANGROVE] / total_class_pixels * 100):.2f}% dari piksel berkelas")
with metric_cols[3]:
    st.metric("Akurasi uji model", "92,76%", help="Nilai dari evaluasi Random Forest yang dilaporkan pada eksperimen. Ini pengujian piksel acak, bukan validasi lapangan independen.")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================

tab_map, tab_stats, tab_about = st.tabs([
    "🗺️ Peta interaktif",
    "📊 Statistik kelas",
    "ℹ️ Tentang proyek",
])

with tab_map:
    left, right = st.columns([3.25, 1], gap="large")

    with left:
        st.markdown('<div class="section-title">Peta tutupan lahan</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Gunakan pilihan kelas untuk menampilkan satu kelas saja, atau pilih semua kelas untuk melihat keseluruhan klasifikasi. Tombol lapisan di peta dapat mengganti basemap satelit dan peta jalan.</div>',
            unsafe_allow_html=True,
        )

        class_options = {"Semua kelas": None}
        for code in CLASS_NAMES:
            class_options[CLASS_LABELS_ID[code]] = code

        selected_label = st.selectbox(
            "Filter tampilan kelas",
            options=list(class_options.keys()),
            index=0,
            key="class_filter",
        )
        selected_code = class_options[selected_label]

        opacity = st.slider(
            "Transparansi lapisan klasifikasi",
            min_value=0.25,
            max_value=1.0,
            value=0.82,
            step=0.05,
        )

        folium_map = build_map(
            class_array=class_array,
            bounds=raster_bounds,
            selected_code=selected_code,
            opacity=opacity,
            show_geojson=show_geojson,
            geojson_obj=geojson_obj,
        )

        st_folium(
            folium_map,
            height=610,
            use_container_width=True,
            returned_objects=[],
            key="landcover_map",
        )

        st.markdown(
            """<div class="small-muted">Catatan: basemap satelit membutuhkan koneksi internet. Lapisan klasifikasi ditampilkan dari raster GeoTIFF yang dipilih.</div>""",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="section-title">Legenda kelas</div>', unsafe_allow_html=True)
        for code in CLASS_NAMES:
            st.markdown(
                f'<div class="legend-pill"><span class="swatch" style="background:{CLASS_COLORS[code]}"></span>{CLASS_LABELS_ID[code]}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Kelas yang dipilih</div>', unsafe_allow_html=True)
        if selected_code is None:
            st.write("Semua kelas ditampilkan bersama.")
        else:
            st.markdown(
                f'<div class="soft-card"><div class="metric-label">{CLASS_LABELS_ID[selected_code]}</div><div class="metric-value">{class_counts[selected_code]:,}</div><div class="metric-note">{(class_counts[selected_code] / total_class_pixels * 100):.2f}% dari piksel berkelas</div></div>',
                unsafe_allow_html=True,
            )
            st.write(CLASS_DESCRIPTIONS[selected_code])

        st.download_button(
            label="⬇️ Unduh raster aktif",
            data=raster_bytes,
            file_name=raster_label or "klasifikasi_jatim.tif",
            mime="image/tiff",
            use_container_width=True,
        )

        st.markdown(
            '<div class="warning-note"><b>Interpretasi peta</b><br>Warna merupakan prediksi model. Terutama untuk Mangrove, hasil tetap perlu dibandingkan dengan citra satelit dan referensi lapangan sebelum dianggap benar sepenuhnya.</div>',
            unsafe_allow_html=True,
        )

with tab_stats:
    st.markdown('<div class="section-title">Distribusi kelas tutupan lahan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Persentase dihitung dari jumlah piksel yang memiliki kode kelas 1–5. Kode 0 tidak dihitung sebagai kelas tutupan lahan.</div>',
        unsafe_allow_html=True,
    )

    chart_df = stats_df.set_index("Kelas")[["Persentase (%)"]]
    st.bar_chart(chart_df, horizontal=True, height=300)

    display_df = stats_df[["Kode", "Kelas", "Nama kelas", "Jumlah piksel", "Persentase (%)"]].copy()
    display_df["Jumlah piksel"] = display_df["Jumlah piksel"].map(lambda x: f"{x:,}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Class-specific details
    chosen_stats_class = st.selectbox(
        "Lihat detail kelas",
        options=list(CLASS_LABELS_ID.values()),
        index=4,
        key="stats_detail_class",
    )
    chosen_code = next(code for code, label in CLASS_LABELS_ID.items() if label == chosen_stats_class)
    st.markdown(
        f'<div class="soft-card"><div class="metric-label">{chosen_stats_class}</div><div class="metric-value">{class_counts[chosen_code]:,} piksel</div><div class="metric-note">{(class_counts[chosen_code] / total_class_pixels * 100):.2f}% dari semua piksel berkelas</div><p style="margin-top:10px;color:#50657d">{CLASS_DESCRIPTIONS[chosen_code]}</p></div>',
        unsafe_allow_html=True,
    )

    st.download_button(
        "⬇️ Unduh tabel statistik CSV",
        data=stats_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="statistik_tutupan_lahan_jatim.csv",
        mime="text/csv",
    )

with tab_about:
    st.markdown('<div class="section-title">Business understanding</div>', unsafe_allow_html=True)
    st.markdown(
        """
        **Tujuan proyek**  
        Membantu pengguna melihat distribusi tutupan lahan pada wilayah kajian di Jawa Timur melalui citra satelit dan klasifikasi berbasis machine learning.

        **Permasalahan**  
        Pemetaan tutupan lahan secara manual membutuhkan waktu dan dapat menghasilkan interpretasi yang berbeda antar-pemetaan. Pembedaan vegetasi hutan dan mangrove juga memerlukan perhatian khusus.

        **Pendekatan solusi**  
        Citra multispektral Sentinel-2 digunakan sebagai masukan untuk model Random Forest. Raster hasil model ditampilkan sebagai lapisan interaktif di atas basemap satelit menggunakan Leaflet.

        **Kelas tutupan lahan**  
        - **Building / Bangunan** — area terbangun.
        - **Farmland / Lahan Pertanian** — lahan pertanian dan budidaya.
        - **Water / Perairan** — laut atau badan air yang diprediksi.
        - **Forest / Hutan** — tutupan hutan selain mangrove.
        - **Mangrove / Hutan Mangrove** — vegetasi mangrove yang diprediksi.

        **Batasan interpretasi**  
        Akurasi uji piksel bukan jaminan akurasi peta pada semua lokasi. Filter pesisir dan MNDWI juga merupakan aturan bantu; area yang terlihat di peta sebaiknya diverifikasi menggunakan citra beresolusi sesuai, data referensi, atau pemeriksaan lapangan.
        """
    )

    st.markdown('<div class="info-note"><b>Catatan penggunaan</b><br>Web ini memvisualisasikan raster klasifikasi yang sudah dibuat. Untuk memperoleh peta yang lebih akurat, perbaikan utama tetap pada kualitas data training, kesesuaian skala band, validasi model, serta pemeriksaan pesisir—bukan sekadar memaksa warna melalui filter.</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    '<div class="small-muted">Jatim LandCover AI · Sentinel-2 L2A · Random Forest · Streamlit + Leaflet</div>',
    unsafe_allow_html=True,
)
