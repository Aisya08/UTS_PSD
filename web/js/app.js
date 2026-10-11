/**
 * WebGIS Tutupan Lahan Jawa Timur (Banyuwangi)
 * Sentinel-2 L2A Multispectral & Random Forest Classification
 * Pure Vanilla JavaScript + Leaflet.js
 */

document.addEventListener("DOMContentLoaded", () => {
  // ============================================================
  // 1. CONFIGURATION & GEOREFERENCED BOUNDS
  // ============================================================
  const CONFIG = {
    // Exact bounding box from GeoTIFF metadata (WGS 84 / EPSG:4326)
    bounds: [
      [-8.6024119, 113.8721975], // South-West [Lat, Lon]
      [-8.1971798, 114.4115945]  // North-East [Lat, Lon]
    ],
    center: [-8.39979585, 114.141896],
    defaultZoom: 10,
    defaultOpacity: 0.85,

    // Classification Layers
    layers: {
      building: {
        name: "Perkotaan / Lahan Terbangun",
        en: "Building",
        code: 1,
        color: "#e53935",
        file: "assets/building.png",
        pixels: 97391,
        pct: "8,21%"
      },
      farmland: {
        name: "Sawah / Lahan Pertanian",
        en: "Farmland",
        code: 2,
        color: "#fbc02d",
        file: "assets/farmland.png",
        pixels: 440701,
        pct: "37,15%"
      },
      water: {
        name: "Badan Air / Perairan",
        en: "Water",
        code: 3,
        color: "#1e88e5",
        file: "assets/water.png",
        pixels: 106089,
        pct: "8,94%"
      },
      forest: {
        name: "Hutan Tropis / Pegunungan",
        en: "Forest",
        code: 4,
        color: "#2e7d32",
        file: "assets/forest.png",
        pixels: 536945,
        pct: "45,26%"
      },
      mangrove: {
        name: "Hutan Mangrove",
        en: "Mangrove",
        code: 5,
        color: "#8e24aa",
        file: "assets/mangrove.png",
        pixels: 5124,
        pct: "0,43%"
      }
    },

    // GeoJSON Vector URL
    roiGeojsonUrl: "data/training_jatim_mangrove_diperkecil_clean.geojson"
  };

  // State variables
  let currentOpacity = CONFIG.defaultOpacity;
  const overlayLayers = {};
  let roiVectorLayer = null;

  // ============================================================
  // 2. INITIALIZE MAP & BASEMAPS
  // ============================================================
  const map = L.map("map", {
    center: CONFIG.center,
    zoom: CONFIG.defaultZoom,
    zoomControl: true,
    attributionControl: true
  });

  // Fit view directly to study area
  map.fitBounds(CONFIG.bounds, { padding: [40, 40] });

  // Basemap Providers
  const basemaps = {
    "esri-sat": L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      {
        attribution: "Tiles &copy; Esri, Maxar, Earthstar Geographics &bull; Copernicus Sentinel-2",
        maxZoom: 18
      }
    ),
    "esri-labeled": L.layerGroup([
      L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        { maxZoom: 18 }
      ),
      L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        {
          attribution: "Labels &copy; Esri",
          maxZoom: 18
        }
      )
    ]),
    "osm": L.tileLayer(
      "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      {
        attribution: "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors",
        maxZoom: 19
      }
    ),
    "carto-dark": L.tileLayer(
      "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
      {
        attribution: "&copy; CARTO &copy; OpenStreetMap",
        maxZoom: 19
      }
    )
  };

  // Add default satellite basemap
  let currentBasemap = basemaps["esri-sat"];
  currentBasemap.addTo(map);

  // Add subtle bounding box boundary line for the study region
  const regionBbox = L.rectangle(CONFIG.bounds, {
    color: "#38bdf8",
    weight: 2,
    fill: false,
    dashArray: "6, 8",
    opacity: 0.85
  }).addTo(map);

  // ============================================================
  // 3. LOAD RASTER CLASSIFICATION OVERLAYS
  // ============================================================
  Object.keys(CONFIG.layers).forEach((key) => {
    const layerMeta = CONFIG.layers[key];

    const overlay = L.imageOverlay(layerMeta.file, CONFIG.bounds, {
      opacity: currentOpacity,
      interactive: false,
      crossOrigin: true
    });

    overlayLayers[key] = overlay;
    overlay.addTo(map); // default checked
  });

  // ============================================================
  // 4. LOAD TRAINING ROI VECTOR LAYER (GEOJSON)
  // ============================================================
  fetch(CONFIG.roiGeojsonUrl)
    .then((res) => {
      if (!res.ok) throw new Error("Gagal membaca berkas GeoJSON");
      return res.json();
    })
    .then((geojsonData) => {
      roiVectorLayer = L.geoJSON(geojsonData, {
        style: (feature) => {
          const className = feature.properties?.class || "";
          let fillColor = "#00e5ff";
          if (className.toLowerCase() === "building") fillColor = CONFIG.layers.building.color;
          else if (className.toLowerCase() === "farmland") fillColor = CONFIG.layers.farmland.color;
          else if (className.toLowerCase() === "water") fillColor = CONFIG.layers.water.color;
          else if (className.toLowerCase() === "forest") fillColor = CONFIG.layers.forest.color;
          else if (className.toLowerCase() === "mangrove") fillColor = CONFIG.layers.mangrove.color;

          return {
            color: "#ffffff",
            weight: 2,
            fillColor: fillColor,
            fillOpacity: 0.65,
            dashArray: "3, 3"
          };
        },
        onEachFeature: (feature, layer) => {
          const cls = feature.properties?.class || "Tidak Terdefinisi";
          layer.bindPopup(`
            <div style="font-family: inherit; font-size: 13px; line-height: 1.5; color: #0b111e;">
              <h4 style="margin: 0 0 6px; color: #1e293b; font-size: 14px; border-bottom: 1px solid #cbd5e1; padding-bottom: 4px;">
                🎯 Poligon Sampel Training (ROI)
              </h4>
              <p style="margin: 3px 0;"><strong>Kelas Tutupan:</strong> <span style="font-weight: 700; color: #2563eb;">${cls}</span></p>
              <p style="margin: 3px 0; font-size: 11px; color: #64748b;">Sampel ground truth yang digunakan untuk ekstraksi fitur multispektral (758 piksel).</p>
            </div>
          `);
        }
      });
      // Do not add to map initially; controlled by #toggle-roi checkbox
    })
    .catch((err) => {
      console.warn("Info: GeoJSON ROI tidak dimuat (opsional):", err);
    });

  // Toggle ROI Checkbox
  const toggleRoi = document.getElementById("toggle-roi");
  if (toggleRoi) {
    toggleRoi.addEventListener("change", (e) => {
      if (!roiVectorLayer) return;
      if (e.target.checked) {
        roiVectorLayer.addTo(map);
        roiVectorLayer.bringToFront();
      } else {
        map.removeLayer(roiVectorLayer);
      }
    });
  }

  // ============================================================
  // 5. LAYER CHECKBOX CONTROLS
  // ============================================================
  const checkboxes = {
    building: document.getElementById("layer-building"),
    farmland: document.getElementById("layer-farmland"),
    water: document.getElementById("layer-water"),
    forest: document.getElementById("layer-forest"),
    mangrove: document.getElementById("layer-mangrove")
  };

  function updateLayerVisibility(key) {
    const chk = checkboxes[key];
    const overlay = overlayLayers[key];
    if (!chk || !overlay) return;

    if (chk.checked) {
      if (!map.hasLayer(overlay)) {
        overlay.setOpacity(currentOpacity);
        overlay.addTo(map);
      }
    } else {
      if (map.hasLayer(overlay)) {
        map.removeLayer(overlay);
      }
    }
  }

  Object.keys(checkboxes).forEach((key) => {
    const chk = checkboxes[key];
    if (chk) {
      chk.addEventListener("change", () => updateLayerVisibility(key));
    }
  });

  // Batch Select All
  const btnCheckAll = document.getElementById("btn-check-all");
  if (btnCheckAll) {
    btnCheckAll.addEventListener("click", () => {
      Object.keys(checkboxes).forEach((key) => {
        checkboxes[key].checked = true;
        updateLayerVisibility(key);
      });
    });
  }

  // Batch Uncheck All
  const btnUncheckAll = document.getElementById("btn-uncheck-all");
  if (btnUncheckAll) {
    btnUncheckAll.addEventListener("click", () => {
      Object.keys(checkboxes).forEach((key) => {
        checkboxes[key].checked = false;
        updateLayerVisibility(key);
      });
    });
  }

  // ============================================================
  // 6. OPACITY SLIDER
  // ============================================================
  const sliderOpacity = document.getElementById("slider-opacity");
  const labelOpacityVal = document.getElementById("label-opacity-val");

  if (sliderOpacity && labelOpacityVal) {
    sliderOpacity.addEventListener("input", (e) => {
      const val = parseInt(e.target.value, 10);
      currentOpacity = val / 100.0;
      labelOpacityVal.textContent = `${val}%`;

      Object.keys(overlayLayers).forEach((key) => {
        const overlay = overlayLayers[key];
        if (map.hasLayer(overlay)) {
          overlay.setOpacity(currentOpacity);
        }
      });
    });
  }

  // ============================================================
  // 7. BASEMAP SWITCHER
  // ============================================================
  const basemapRadios = document.querySelectorAll('input[name="basemap-choice"]');
  basemapRadios.forEach((radio) => {
    radio.addEventListener("change", (e) => {
      const choice = e.target.value;
      if (basemaps[choice]) {
        map.removeLayer(currentBasemap);
        currentBasemap = basemaps[choice];
        currentBasemap.addTo(map);

        // Bring overlays to front
        Object.keys(overlayLayers).forEach((key) => {
          const overlay = overlayLayers[key];
          if (map.hasLayer(overlay)) {
            overlay.bringToFront();
          }
        });
        if (roiVectorLayer && map.hasLayer(roiVectorLayer)) {
          roiVectorLayer.bringToFront();
        }
        if (regionBbox) regionBbox.bringToFront();
      }
    });
  });

  // ============================================================
  // 8. MAP RECENTER & COORDINATES TRACKER
  // ============================================================
  const btnRecenter = document.getElementById("btn-recenter");
  if (btnRecenter) {
    btnRecenter.addEventListener("click", () => {
      map.fitBounds(CONFIG.bounds, { padding: [40, 40] });
    });
  }

  const valLat = document.getElementById("val-lat");
  const valLon = document.getElementById("val-lon");
  const valZoom = document.getElementById("val-zoom");

  map.on("mousemove", (e) => {
    if (valLat) valLat.textContent = e.latlng.lat.toFixed(4);
    if (valLon) valLon.textContent = e.latlng.lng.toFixed(4);
  });

  map.on("zoomend", () => {
    if (valZoom) valZoom.textContent = map.getZoom();
  });

  // Map Click Interactive Popup
  map.on("click", (e) => {
    const lat = e.latlng.lat;
    const lon = e.latlng.lng;

    // Check if click is inside study bounding box
    const inBounds =
      lat >= CONFIG.bounds[0][0] &&
      lat <= CONFIG.bounds[1][0] &&
      lon >= CONFIG.bounds[0][1] &&
      lon <= CONFIG.bounds[1][1];

    const content = `
      <div style="font-family: inherit; font-size: 12px; line-height: 1.5; color: #1e293b;">
        <strong style="color: #0284c7; font-size: 13px;">📍 Koordinat Wilayah</strong><br>
        <strong>Latitude:</strong> ${lat.toFixed(6)}&deg; S<br>
        <strong>Longitude:</strong> ${lon.toFixed(6)}&deg; E<br>
        <span style="display:inline-block; margin-top:4px; padding:2px 6px; border-radius:4px; font-size:11px; font-weight:600; background:${
          inBounds ? "#dcfce7; color:#15803d" : "#fee2e2; color:#b91c1c"
        };">
          ${inBounds ? "✓ Didalam Wilayah Studi Banyuwangi" : "✕ Diluar Bounding Box Citra"}
        </span>
      </div>
    `;

    L.popup()
      .setLatLng(e.latlng)
      .setContent(content)
      .openOn(map);
  });

  // ============================================================
  // 9. PANEL TABS NAVIGATION & TOGGLE
  // ============================================================
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");

      tabButtons.forEach((b) => b.classList.remove("active"));
      tabContents.forEach((c) => c.classList.remove("active"));

      btn.classList.add("active");
      const targetContent = document.getElementById(targetId);
      if (targetContent) targetContent.classList.add("active");
    });
  });

  const sidebarPanel = document.getElementById("sidebar-panel");
  const btnToggleSidebar = document.getElementById("btn-toggle-sidebar");

  if (btnToggleSidebar && sidebarPanel) {
    btnToggleSidebar.addEventListener("click", () => {
      sidebarPanel.classList.toggle("collapsed");
    });
  }

  // ============================================================
  // 10. MODAL DIALOG
  // ============================================================
  const btnInfoModal = document.getElementById("btn-info-modal");
  const infoModal = document.getElementById("info-modal");
  const btnCloseModal = document.getElementById("btn-close-modal");

  if (btnInfoModal && infoModal) {
    btnInfoModal.addEventListener("click", () => {
      infoModal.classList.remove("hidden");
    });
  }

  if (btnCloseModal && infoModal) {
    btnCloseModal.addEventListener("click", () => {
      infoModal.classList.add("hidden");
    });
  }

  if (infoModal) {
    infoModal.addEventListener("click", (e) => {
      if (e.target === infoModal) {
        infoModal.classList.add("hidden");
      }
    });
  }
});
