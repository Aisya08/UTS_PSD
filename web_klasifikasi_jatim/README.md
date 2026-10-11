# Jatim LandCover AI — Streamlit + Leaflet

Web dashboard untuk menampilkan raster klasifikasi tutupan lahan berbasis Sentinel-2 dengan lima kelas:

1. Building (kode 1)
2. Farmland (kode 2)
3. Water (kode 3)
4. Forest (kode 4)
5. Mangrove (kode 5)

Fitur: peta Leaflet dengan basemap satelit dan peta jalan, filter tampilan satu kelas, transparansi overlay, statistik dan persentase kelas, ekspor CSV, unduh GeoTIFF, penjelasan business understanding, dan opsi overlay GeoJSON training.

## Menjalankan secara lokal

Disarankan menggunakan Python 3.11.

```powershell
cd D:\semester 5\psd\klasifikasi_jatim\web_klasifikasi_jatim
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Kalau folder web berada di dalam folder proyek yang sudah memuat raster, opsi **File di folder aplikasi** akan mendeteksi file `hasil_klasifikasi_jatim*.tif` atau `.tiff`. Web memprioritaskan `hasil_klasifikasi_jatim_FINAL.tif` jika tersedia.

Jika raster berada di folder lain, pilih **Unggah GeoTIFF** di sidebar dan unggah file hasil klasifikasi. Raster harus satu band dengan kode kelas 1 sampai 5; kode 0 diperlakukan sebagai latar belakang/NoData.

Untuk menampilkan polygon training, aktifkan **Tampilkan polygon training** di sidebar. Web akan mencari file GeoJSON training yang dikenal di folder aplikasi atau memberi pilihan untuk mengunggah GeoJSON.

## Deploy ke Streamlit Community Cloud

1. Buat repository GitHub berisi `app.py`, `requirements.txt`, dan file GeoTIFF hasil klasifikasi yang akan dipakai secara default (misalnya `hasil_klasifikasi_jatim_FINAL.tif`).
2. Jika perlu overlay polygon training, sertakan `training_jatim_mangrove_diperkecil_clean.geojson` atau gunakan opsi upload.
3. Di Streamlit Community Cloud, pilih repository dan file `app.py` sebagai entry point.
4. Set Python ke 3.11 jika tersedia pada pengaturan deployment.

**Catatan:** jangan unggah data sensitif. Untuk raster besar, lebih praktis menggunakan opsi upload atau penyimpanan yang sesuai. Peta basemap satelit membutuhkan koneksi internet.

## Catatan hasil ilmiah

Web menampilkan kelas yang sudah ada dalam raster; ia tidak melatih model dari awal. Nilai akurasi 92,76% adalah angka evaluasi eksperimen pada pixel testing acak, bukan validasi lapangan yang independen. Prediksi Mangrove perlu ditinjau terhadap citra dan data referensi.
