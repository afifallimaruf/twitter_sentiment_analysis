# **Sentiment Analysis Social Media(Twitter dataset)✨**

#### Proyek ini adalah aplikasi analisis sentimen berbasis web yang menganalisis teks dari postingan di media sosial. Dalam aplikasi ini user memasukan teks postingan pada media sosial di dalam form yang dibuat dengan React js, setelah form terisi dan user menekan tombol 'Analyze Sentiment' teks yang di dalam form dikirim ke sebuah API yang dibuat dengan Flask. di dalam API, teks tadi di proses dan dimasukan ke dalam model yang telah dilatih untuk menentukan apakah teks tadi memiliki sentimen positif atau negatif. hasil dari API ini(Positive/Negative) akan dikembalikan lagi ke aplikasi frontend yang nantinya dalam aplikasi frontend/website muncul apakah teks yang di submit memiliki sentimen positif atau negatif.

# **Fitur**

- Frontend: Antarmuka web yang digunakan menggunakan React untuk memasukan dan menampilkan hasil analisis sentimen (positif/negatif)
- Backend(API): Memproses teks menggunakan model machine learning yang telah dilatih.
- Model: Model Logistic Regression dengan TfidfVectorizer untuk analisis sentimen
- Dataset: Menggunakan dataset [Sentiment140 dari kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
- MLOps: Pelacakan eksperimen menggunakan MLflow(lokal)

# **Prasyarat**

- Python 3.9+ untuk backend dan pelatihan model
- Node.js 16+ dan npm untuk frontend
- Git untuk mengelola repositori
- Dataset Sentiment140: Unduh dari kaggle dan letakan di folder data/raw/sentiment140.csv

# **Instalasi(lokal)**

1. Kloning Repositori:

<<<<<<< HEAD
   ```bash
   git clone <your-repo-url>
   cd twitter_sentiment_analysis
   ```

   ```bash
   git clone [https://github.com/user/repo.git](https://github.com/user/repo.git)
   ```
=======
   `git clone <your-repo-url>
   cd twitter_sentiment_analysis`
>>>>>>> 71c19c91fdce82a3c66199d44f2869fce671caa2

2. Siapkan Backend:

- Buat dan aktifkan virtual environment:

  `python3 -m venv <nama-virtual-environment>
   source <nama-virtual-environment>/bin/activate`

- Install dependensi:

  `pip install -r requirements.txt`

3. Latih Model (jika belum ada model.pkl):

- Pastikan dataset data/raw/sentiment140.csv tersedia.
- Jalankan:

  `python3 src/train_model.py`

4. Jalankan Mlflow (Opsional):

- Untuk pelacakan eksperimen:

  `mlflow server --port 5001`

5. Siapkan Frontend:

- Masuk ke direktori frontend:

  `cd frontend
   npm install`

- Jalankan sever pengembangan:

  `npm run dev`

6. Jalankan Backend:

- di direktori lain (api), aktifkan virtual environment dan jalankan:

  `cd api
   python3 app.py`
