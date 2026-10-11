# Jatim LandCover Map 🌍🛰️

**Peta Klasifikasi Tutupan Lahan Jawa Timur Berdasarkan Citra Satelit Sentinel-2 dan Algoritma Random Forest**

Website statis interaktif berbasis **HTML, CSS, JavaScript, dan Leaflet** untuk menampilkan hasil klasifikasi tutupan lahan di wilayah Jawa Timur (area Banyuwangi dan sekitarnya). Website ini 100% statis tanpa ketergantungan pada backend, database, maupun Streamlit, sehingga sangat ringan, cepat, dan siap dipublikasikan langsung ke **GitHub Pages** atau dijalankan menggunakan **VS Code Live Server**.

---

## 🎨 Fitur Utama

1. **Peta Satelit Interaktif**:
   - Menampilkan basemap citra satelit resolusi tinggi (**Esri World Imagery**) sebagai peta dasar default.
   - Pilihan basemap tambahan: *Satelit + Label Batas Wilayah* dan *OpenStreetMap*.
   - Navigasi penuh: Zoom in/out, panning/geser, dan tombol *Fokus Wilayah*.

2. **Overlay Klasifikasi Georeferensi Akurat**:
   - Menggunakan koordinat batas (*bounding box*) nyata dari GeoTIFF hasil klasifikasi:
     - **Batas Barat (West)**: `113.8721975° E`
     - **Batas Timur (East)**: `114.4115945° E`
     - **Batas Selatan (South)**: `-8.6024119° S`
     - **Batas Utara (North)**: `-8.1971798° S`
     - **Sistem Koordinat**: `EPSG:4326 (WGS 84)`

3. **5 Kategori Tutupan Lahan dengan Checkbox Mandiri**:
   - 🟥 **Perkotaan / Lahan Terbangun (*Building*)**
   - 🟨 **Sawah / Lahan Pertanian (*Farmland*)**
   - 🟦 **Perairan (*Water*)**
   - 🟩 **Hutan (*Forest*)**
   - 🟪 **Hutan Mangrove (*Mangrove*)**
   - Mendukung pemilihan multi-kategori (misal: aktifkan Perkotaan dan Sawah secara bersamaan).
   - Menghilangkan layer secara bersih ketika checkbox dinonaktifkan tanpa menimbulkan layer duplikat.
   - Tombol instan *Pilih Semua* dan *Sembunyikan Semua*.

4. **Pengatur Transparansi (*Opacity Slider*)**:
   - Memungkinkan pengguna mengatur transparansi overlay (10% - 100%) secara *real-time* untuk membandingkan hasil klasifikasi dengan citra satelit di bawahnya.

5. **Desain Modern & Responsif**:
   - Tema elegan: **Biru Langit & Putih Modern** dengan efek *glassmorphism*.
   - Kompatibel dan responsif di layar komputer, tablet, maupun smartphone.

---

## 📁 Struktur Berkas

```
web_statis_jatim/
│
├── index.html        # Struktur antarmuka dan container peta
├── style.css         # Desain tema biru langit & putih, responsif & glassmorphism
├── script.js         # Logika interaktif Leaflet, kontrol checkbox, opacity & basemap
├── prepare_map.py    # Script konversi raster GeoTIFF menjadi PNG transparan
├── README.md         # Dokumentasi lengkap proyek
│
└── assets/           # Berkas gambar overlay dan metadata
    ├── bounds.json   # Metadata bounds koordinat dan statistik piksel
    ├── building.png  # Layer PNG transparan kelas Perkotaan (Merah)
    ├── farmland.png  # Layer PNG transparan kelas Sawah (Kuning)
    ├── water.png     # Layer PNG transparan kelas Perairan (Biru)
    ├── forest.png    # Layer PNG transparan kelas Hutan (Hijau)
    └── mangrove.png  # Layer PNG transparan kelas Mangrove (Ungu)
```

---

## 🚀 Panduan Menjalankan Website

### Opsi 1: Menggunakan VS Code Live Server (Direkomendasikan)
1. Buka folder `web_statis_jatim` di Visual Studio Code.
2. Pasang ekstensi **Live Server** (oleh Ritwick Dey) jika belum terpasang.
3. Klik kanan pada berkas `index.html`, lalu pilih **"Open with Live Server"**.
4. Website akan terbuka di peramban pada alamat `http://127.0.0.1:5500/index.html`.

### Opsi 2: Menggunakan Python HTTP Server
Buka terminal/PowerShell di dalam folder `web_statis_jatim`:
```bash
python -m http.server 8000
```
Buka peramban dan akses: [http://localhost:8000](http://localhost:8000)

### Opsi 3: Publikasi ke GitHub Pages
1. Unggah seluruh isi folder `web_statis_jatim` ke repository GitHub Anda.
2. Buka menu **Settings** > **Pages** pada repository GitHub.
3. Pada bagian **Build and deployment**, pilih branch `main` (atau `master`) dan folder `/ (root)`.
4. Simpan, dan website statis Anda akan langsung online dan dapat diakses publik.

> **Catatan**: Basemap citra satelit memerlukan koneksi internet aktif untuk memuat tile peta dari Esri/OpenStreetMap.

---

## 🛠️ Menyiapkan / Memperbarui PNG dari Raster (`prepare_map.py`)

Jika Anda memperbarui raster klasifikasi GeoTIFF, jalankan kembali script `prepare_map.py` untuk mengonversi raster ke berkas PNG baru di folder `assets/`.

### Urutan Prioritas Otomatis:
Script akan otomatis mencari berkas raster dengan urutan prioritas:
1. `hasil_klasifikasi_jatim_FINAL.tif`
2. `hasil_klasifikasi_jatim_mangrove_koridor_pesisir.tif`
3. `hasil_klasifikasi_jatim.tif`

### Cara Menjalankan:
Pastikan modul Python `rasterio`, `numpy`, dan `Pillow` sudah terpasang:
```bash
pip install rasterio numpy Pillow
```

Jalankan script:
```bash
python prepare_map.py
```

Script akan:
1. Membaca raster klasifikasi dan metadata batas georeferensi.
2. Menghasilkan 5 berkas PNG transparan (satu untuk setiap kelas tutupan lahan).
3. Menyimpan metadata batas dan persentase luas ke `assets/bounds.json`.

---

## 📊 Statistik Tutupan Lahan (Data Final)

| Kode | Kategori | Warna | Jumlah Piksel | Persentase |
|:---:|:---|:---:|:---:|:---:|
| **1** | Perkotaan / Lahan Terbangun (*Building*) | 🟥 Merah (`#E53935`) | 167.076 | 14.08% |
| **2** | Sawah / Lahan Pertanian (*Farmland*) | 🟨 Kuning (`#FBC02D`) | 563.903 | 47.54% |
| **3** | Perairan (*Water*) | 🟦 Biru (`#1E88E5`) | 81.341 | 6.86% |
| **4** | Hutan (*Forest*) | 🟩 Hijau (`#2E7D32`) | 361.543 | 30.48% |
| **5** | Hutan Mangrove (*Mangrove*) | 🟪 Ungu (`#8E24AA`) | 12.387 | 1.04% |
| **Total** | | | **1.186.250** | **100.00%** |

---

## 👥 Pengembang Proyek
- **Dataset**: Citra Satelit Sentinel-2 L2A (Kanal B04, B08, B8A, B11 & indeks MNDWI/NDVI)
- **Metode**: Supervised Classification — Random Forest Classifier
- **Wilayah Studi**: Jawa Timur (Kawasan Pesisir & Daratan Banyuwangi)
