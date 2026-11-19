# Klasifikasi Citra Penyakit Daun Tomat (PlantVillage)

Perbandingan Simple CNN, Modified CNN, dan InceptionV3 pada 6 kelas daun tomat
dengan 5-fold stratified cross validation. Kelompok 6, S1 Sains Data, Telkom University Surabaya.

## Struktur
| File | Isi |
|---|---|
| `preprocessing.py` | Konfigurasi, pemuatan data, resize/rescale/augmentasi |
| `model.py` | Arsitektur Algoritma 1 (Simple CNN), 2 (Modified CNN), 3 (InceptionV3) |
| `train.py` | Training 5-fold, menyimpan `hasil_algoN.json` |
| `test.py` | Evaluasi: tabel per fold, rata-rata, confusion matrix, F1 per kelas |

## Cara menjalankan
```bash
pip install -r requirements.txt
# unduh dataset (butuh akun & token Kaggle)
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip -q plantvillage-dataset.zip "plantvillage dataset/color/Tomato___*" -d dataset_daun/
python train.py      # latih 3 algoritma (disarankan GPU)
python test.py       # hasilkan tabel & gambar
```
