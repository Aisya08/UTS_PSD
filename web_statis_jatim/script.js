/**
 * Jatim LandCover Map - Interactive Static Web Application
 * Sentinel-2 Land Cover Classification using Leaflet
 */

// ============================================================
// 1. DATA CONFIGURATION & BOUNDS (From GeoTIFF Metadata)
// ============================================================

const CONFIG = {
  // Koordinat bounds georeferensi nyata raster Jawa Timur
  bounds: [
    [-8.6024119, 113.8721975], // South-West [Lat, Lon]
    [-8.1971798, 114.4115945]  // North-East [Lat, Lon]
  ],
  center: [-8.39979585, 114.141896],
  defaultZoom: 10,
  defaultOpacity: 0.85,

  // Metadata 5 Kelas Tutupan Lahan
  layers: {
    building: {
      name: "Perkotaan / Lahan Terbangun",
      file: "assets/building.png",
      color: "#E53935",
      defaultVisible: true
    },
    farmland: {
      name: "Sawah / Lahan Pertanian",
      file: "assets/farmland.png",
      color: "#FBC02D",
      defaultVisible: true
    },
    water: {
      name: "Perairan",
      file: "assets/water.png",
      color: "#1E88E5",
      defaultVisible: true
    },
    forest: {
      name: "Hutan",
      file: "assets/forest.png",
      color: "#2E7D32",
      defaultVisible: true
    },
    mangrove: {
      name: "Hutan Mangrove",
      file: "assets/mangrove.png",
      color: "#8E24AA",
      defaultVisible: true
    }
  }
};

// ============================================================
// 2. INITIALIZE MAP & BASEMAPS
// ============================================================

// Initialize Leaflet Map
const map = L.map("map", {
  center: CONFIG.center,
  zoom: CONFIG.defaultZoom,
  zoomControl: true,
  attributionControl: true
});

// Fit map into the exact raster bounds
map.fitBounds(CONFIG.bounds, { padding: [40, 40] });

// Basemap Tile Providers
const basemaps = {
  "esri-sat": L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    {
      attribution: "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
      maxZoom: 18
    }
  ),
  "esri-clarity": L.layerGroup([
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
  )
};

// Add default satellite basemap
let currentBasemap = basemaps["esri-sat"];
currentBasemap.addTo(map);

// Add raster extent boundary rectangle (subtle boundary indicator)
const rasterBBox = L.rectangle(CONFIG.bounds, {
  color: "#38bdf8",
  weight: 1.5,
  fill: false,
  dashArray: "4, 6",
  opacity: 0.8
}).addTo(map);

// ============================================================
// 3. CLASSIFICATION OVERLAYS MANAGEMENT
// ============================================================

// Store Leaflet imageOverlay instances
const overlayLayers = {};
let currentOpacity = CONFIG.defaultOpacity;

// Create Image Overlay for each category
Object.keys(CONFIG.layers).forEach((key) => {
  const layerInfo = CONFIG.layers[key];

  const overlay = L.imageOverlay(layerInfo.file, CONFIG.bounds, {
    opacity: currentOpacity,
    interactive: false,
    crossOrigin: true
  });

  overlayLayers[key] = overlay;

  // Add to map by default if set
  if (layerInfo.defaultVisible) {
    overlay.addTo(map);
  }
});

// ============================================================
// 4. CHECKBOX EVENT HANDLERS
// ============================================================

const checkboxes = {
  building: document.getElementById("chk-building"),
  farmland: document.getElementById("chk-farmland"),
  water: document.getElementById("chk-water"),
  forest: document.getElementById("chk-forest"),
  mangrove: document.getElementById("chk-mangrove")
};

/**
 * Update layer visibility based on checkbox state.
 * Fully supports individual toggling and multiple active layers simultaneously.
 */
function updateLayerVisibility(key) {
  const checkbox = checkboxes[key];
  const overlay = overlayLayers[key];

  if (!checkbox || !overlay) return;

  if (checkbox.checked) {
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

// Bind change listeners to all checkboxes
Object.keys(checkboxes).forEach((key) => {
  const chk = checkboxes[key];
  if (chk) {
    chk.addEventListener("change", () => {
      updateLayerVisibility(key);
    });
  }
});

// Select All Button Handler
const btnSelectAll = document.getElementById("btn-select-all");
if (btnSelectAll) {
  btnSelectAll.addEventListener("click", () => {
    Object.keys(checkboxes).forEach((key) => {
      checkboxes[key].checked = true;
      updateLayerVisibility(key);
    });
  });
}

// Clear All Button Handler
const btnClearAll = document.getElementById("btn-clear-all");
if (btnClearAll) {
  btnClearAll.addEventListener("click", () => {
    Object.keys(checkboxes).forEach((key) => {
      checkboxes[key].checked = false;
      updateLayerVisibility(key);
    });
  });
}

// ============================================================
// 5. OPACITY SLIDER
// ============================================================

const opacitySlider = document.getElementById("layer-opacity");
const opacityValue = document.getElementById("opacity-value");

if (opacitySlider && opacityValue) {
  opacitySlider.addEventListener("input", (e) => {
    const val = parseInt(e.target.value, 10);
    currentOpacity = val / 100.0;
    opacityValue.textContent = `${val}%`;

    // Apply new opacity to all currently active overlay layers
    Object.keys(overlayLayers).forEach((key) => {
      const overlay = overlayLayers[key];
      if (map.hasLayer(overlay)) {
        overlay.setOpacity(currentOpacity);
      }
    });
  });
}

// ============================================================
// 6. BASEMAP SWITCHER
// ============================================================

const basemapRadios = document.querySelectorAll('input[name="basemap"]');
basemapRadios.forEach((radio) => {
  radio.addEventListener("change", (e) => {
    const selectedKey = e.target.value;
    if (basemaps[selectedKey]) {
      map.removeLayer(currentBasemap);
      currentBasemap = basemaps[selectedKey];
      currentBasemap.addTo(map);
      
      // Ensure overlays stay on top of new basemap
      Object.keys(overlayLayers).forEach((key) => {
        const overlay = overlayLayers[key];
        if (map.hasLayer(overlay)) {
          overlay.bringToFront();
        }
      });
      if (rasterBBox) rasterBBox.bringToFront();
    }
  });
});

// ============================================================
// 7. HEADER ACTIONS & UI CONTROLS
// ============================================================

// Recenter Button
const btnRecenter = document.getElementById("btn-recenter");
if (btnRecenter) {
  btnRecenter.addEventListener("click", () => {
    map.flyToBounds(CONFIG.bounds, {
      padding: [40, 40],
      duration: 1.2
    });
  });
}

// Toggle Floating Panel
const controlPanel = document.getElementById("control-panel");
const btnTogglePanel = document.getElementById("btn-toggle-panel");
const btnClosePanel = document.getElementById("btn-close-panel");

if (btnTogglePanel && controlPanel) {
  btnTogglePanel.addEventListener("click", () => {
    controlPanel.classList.toggle("collapsed");
  });
}

if (btnClosePanel && controlPanel) {
  btnClosePanel.addEventListener("click", () => {
    controlPanel.classList.add("collapsed");
  });
}

// Coordinates & Zoom Display
const coordsDisplay = document.getElementById("map-coordinates");

function updateCoords(lat, lng, zoom) {
  if (coordsDisplay) {
    coordsDisplay.textContent = `Lat: ${lat.toFixed(4)} | Lon: ${lng.toFixed(4)} | Zoom: ${zoom}`;
  }
}

map.on("mousemove", (e) => {
  updateCoords(e.latlng.lat, e.latlng.lng, map.getZoom());
});

map.on("zoomend", () => {
  const center = map.getCenter();
  updateCoords(center.lat, center.lng, map.getZoom());
});

// Initial coords display
updateCoords(CONFIG.center[0], CONFIG.center[1], CONFIG.defaultZoom);

console.log("Jatim LandCover Map loaded successfully.");
