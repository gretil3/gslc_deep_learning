# Tugas Take-Home Sesi 05: Transfer Learning untuk Klasifikasi Citra dengan Data Terbatas

**Nama:** David Gaby Sahala Sinambela  
**NIM:** 2802520293  
**Mata kuliah:** Deep Learning (Sesi 05: Advanced CNNs and Transfer Learning)

Repositori ini berisi pengerjaan tugas take-home Sesi 05. Tujuannya membandingkan beberapa strategi pelatihan CNN untuk klasifikasi ras kucing dan anjing dengan data berlabel yang terbatas: pelatihan dari nol, feature extraction, dan progressive fine-tuning, ditambah uji kontrol data augmentation dan perbandingan dengan EfficientNet-B0.

## Isi Repositori

| File / Folder | Isi |
|---|---|
| `Part A - Analisis Konseptual.pdf` | Bagian A: jawaban lima pertanyaan konseptual |
| `Part B - Experiment.ipynb` | Bagian B: notebook eksperimen B1 sampai B5 yang sudah dieksekusi lengkap beserta output-nya |
| `Part C - Laporan dan Analisis.pdf` | Bagian C: laporan analisis (C1 sampai C4) dan pernyataan penggunaan AI |
| `logs/` | Log metrik hasil eksekusi notebook (bukti integritas data) |
| `requirements.txt` | Library dan versi yang dipakai |

Isi folder `logs/`:

| File | Isi |
|---|---|
| `results.csv` | Ringkasan metrik tiap kondisi (6 baris: B1, B2, B3, B4_noaug, B4_aug, B5): trainable parameters, learning rate per kelompok parameter, epoch konvergen, jumlah epoch yang dijalankan, waktu latih, val loss terbaik, akurasi validasi, dan Macro F1 test. Untuk B4 juga ada akurasi train bersih dan generalization gap |
| `history.csv` | Loss dan akurasi train/validasi per epoch untuk semua kondisi (83 baris), lengkap dengan learning rate, waktu per epoch, dan penanda epoch terbaik |

Catatan: kedua file ini diekstrak dari output sel notebook `Part B - Experiment.ipynb`, jadi presisinya mengikuti angka yang tercetak (4 desimal untuk loss/akurasi, 1 desimal untuk waktu per epoch). Kolom `train_acc_clean` dan `gap_train_val` hanya terisi untuk B4, karena hanya nilai itu yang ditampilkan di notebook. Saat dijalankan, notebook juga menyimpan log lengkap (termasuk `test_predictions.csv` dan checkpoint `.pt`) ke Google Drive di `MyDrive/tugas05_outputs/`.

## Dataset

Dataset yang dipakai adalah **Oxford-IIIT Pet**, diunduh otomatis lewat `torchvision.datasets.OxfordIIITPet` (tidak perlu autentikasi). Dari 37 kelas, diambil 10 kelas:

- **Kucing:** Abyssinian, Birman, British Shorthair, Ragdoll, Russian Blue
- **Anjing:** American Pit Bull Terrier, Beagle, Samoyed, Shiba Inu, Staffordshire Bull Terrier

Split `trainval` dan `test` bawaan torchvision digabung dulu, lalu dibagi ulang secara stratified menjadi 70/15/15: **1.390 citra latih, 298 citra validasi, dan 299 citra uji** (total 1.987 citra). Tiga pasangan ras yang mirip (Birman–Ragdoll, British Shorthair–Russian Blue, Pit Bull–Staffordshire) sengaja dimasukkan supaya failure analysis punya kasus yang bermakna.

## Protokol Eksperimen

- Arsitektur utama ResNet-18 (`ResNet18_Weights.IMAGENET1K_V1`), input 224×224 dengan normalisasi ImageNet
- Batch size 32, optimizer AdamW (weight decay 0,01), CrossEntropyLoss
- Early stopping pada validation loss (patience 5, maksimal 30 epoch); bobot dengan val loss terbaik yang dipakai untuk evaluasi test
- Seed 42 untuk Python, NumPy, dan PyTorch, dengan `cudnn.deterministic = True`
- Augmentasi (RandomResizedCrop, horizontal flip, rotasi 15°, color jitter brightness/contrast) hanya diterapkan pada data latih

| Kode | Kondisi | Keterangan |
|---|---|---|
| B1 | Baseline from scratch | ResNet-18 dari inisialisasi acak, semua layer dilatih (LR 1e-3) |
| B2 | Feature extraction | Backbone dibekukan, hanya `fc` baru yang dilatih (LR 1e-3) |
| B3 | Progressive fine-tuning | `layer4` + `fc` dibuka dengan differential learning rate (1e-4 / 1e-3) |
| B4 | Uji kontrol augmentasi | Konfigurasi B3 tanpa augmentasi (`B4_noaug`) dan dengan augmentasi (`B4_aug`) |
| B5 | Arsitektur pembanding (bonus) | EfficientNet-B0 dengan konfigurasi terbaik berdasarkan akurasi validasi |

## Cara Menjalankan

1. Buka `Part B - Experiment.ipynb` di [Google Colab](https://colab.research.google.com/).
2. Pilih **Runtime → Change runtime type → T4 GPU**.
3. Pilih **Runtime → Run all**, lalu izinkan akses Google Drive saat diminta. Semua log dan grafik akan tersimpan di `MyDrive/tugas05_outputs/`.
4. Kalau runtime terputus di tengah jalan, ubah `RESUME = True` di sel konfigurasi lalu jalankan ulang. Kondisi yang sudah selesai akan dimuat dari Drive, jadi tidak dilatih ulang.

Untuk menjalankan di komputer sendiri (butuh GPU CUDA), install library dengan:

```bash
pip install -r requirements.txt
```

lalu set `USE_DRIVE = False` di notebook, supaya output tersimpan di folder lokal `outputs/`.

## Ringkasan Hasil

Angka di bawah disalin dari `logs/results.csv` (hasil satu kali eksekusi dengan seed 42 di GPU Tesla T4).

| Kode | Strategi | Trainable Params | Epoch Konvergen | Waktu Latih (s) | Val Acc | Test Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| B1 | From scratch | 11.181.642 | 15 | 341.2 | 0.4933 | 0.4335 |
| B2 | Feature extraction | 5.130 | 21 | 379.3 | 0.8758 | 0.9066 |
| B3 | Progressive fine-tuning | 8.398.858 | 3 | 118.5 | 0.8893 | 0.9000 |
| B4_noaug | B3 tanpa augmentasi | 8.398.858 | 4 | 95.6 | 0.9094 | 0.9156 |
| B4_aug | B3 dengan augmentasi | 8.398.858 | 3 | 120.0 | 0.8893 | 0.9000 |
| B5 | Fine-tuning EfficientNet-B0 | 3.168.550 | 7 | 130.5 | 0.9329 | **0.9423** |

Poin utamanya:

- Transfer learning (B2/B3) unggul jauh dari pelatihan dari nol (B1), dengan selisih Macro F1 test sekitar 47 poin.
- B2 dan B3 punya akurasi yang hampir setara, tetapi B3 konvergen jauh lebih cepat (epoch 3 vs epoch 21).
- Pada eksperimen ini augmentasi hanya sedikit menurunkan generalization gap (0.0906 menjadi 0.0856) dan tidak menaikkan akurasi.
- EfficientNet-B0 (B5) memberi Macro F1 test tertinggi dengan total parameter sekitar 64% lebih sedikit dari ResNet-18.
- Pada model terbaik (B5), 9 dari 17 kesalahan di test set terjadi antara American Pit Bull Terrier dan Staffordshire Bull Terrier.

Pembahasan lengkapnya ada di `Part C - Laporan dan Analisis.pdf`.

## Reproduksibilitas

Seed dikunci dan `cudnn.deterministic` dinyalakan. Kondisi `B4_aug` memakai konfigurasi yang sama dengan B3, dan hasilnya identik di setiap epoch, jadi pipeline-nya terbukti deterministik. Hasil hanya diuji dengan satu seed, jadi variasi antar-seed belum terukur. Waktu latih diukur di GPU Colab yang dipakai bersama, jadi angkanya bisa sedikit berbeda kalau dijalankan ulang.

## Pernyataan Penggunaan AI

Tugas ini dikerjakan dengan bantuan AI (Claude dan Gemini). Rincian ruang lingkup bantuannya dan cara verifikasinya ada di lampiran **Pernyataan Penggunaan AI** pada laporan Part C.

## Referensi

- O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar, "Cats and dogs," in *Proc. IEEE CVPR*, 2012. https://www.robots.ox.ac.uk/~vgg/data/pets/
- PyTorch, "torchvision.datasets.OxfordIIITPet," torchvision documentation. https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html
- PyTorch, "Models and pre-trained weights," torchvision documentation. https://docs.pytorch.org/vision/stable/models.html
- K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE CVPR*, 2016, pp. 770–778. https://arxiv.org/abs/1512.03385
- M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. ICML*, PMLR vol. 97, 2019, pp. 6105–6114. https://proceedings.mlr.press/v97/tan19a.html
- Hidayaturrahman, "Session 05: Advanced CNNs and Transfer Learning," materi kuliah Deep Learning (file PPT kelas).
