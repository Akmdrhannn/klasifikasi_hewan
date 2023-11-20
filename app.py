import pickle
import streamlit as st
import pickle
import numpy as np

# Dictionary untuk mapping tipe ke nama
tipe_to_nama = {
    1: "Mamalia",
    2: "Burung/Unggas",
    3: "Reptil",
    4: "Ikan",
    5: "Amfibi",
    6: "Serangga",
    7: "Moluska/Binatang Laut"
}

# Dictionary untuk mapping fitur kategorikal ke arti
arti_fitur = {
    'hair': 'Memiliki rambut',
    'feathers': 'Memiliki bulu',
    'eggs': 'Bertelur',
    'milk': 'Menyusui',
    'airborne': 'Dapat terbang',
    'aquatic': 'Hidup di air',
    'predator': 'Pemangsa',
    'toothed': 'Memiliki gigi',
    'backbone': 'Memiliki tulang belakang',
    'breathes': 'Bernapas',
    'venomous': 'Beracun',
    'fins': 'Memiliki sirip',
    'tail': 'Memiliki ekor',
    'domestic': 'Dapat dijinakkan',
    'catsize': 'Berukuran seperti kucing'
}

# Muat model .pkl
with open('svm_model.pkl', 'rb') as model_file:
    loaded_object = pickle.load(model_file)

# Periksa apakah objek yang dimuat adalah model
if hasattr(loaded_object, 'predict'):
    model = loaded_object
else:
    st.error("Error: Objek yang dimuat tidak memiliki metode 'predict'. Mohon periksa model yang disimpan.")

# Judul Aplikasi
st.title('Aplikasi Klasifikasi Hewan')

# Nama Hewan
nama_hewan = st.text_input('Nama Hewan')
# Formulir Input Streamlit di bawah judul
st.header('Masukkan Fitur Hewan')

# List nama fitur kategorikal
fitur_kategorikal = ['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic',
                     'predator', 'toothed', 'backbone', 'breathes', 'venomous',
                     'fins', 'tail', 'domestic', 'catsize']

# Menyiapkan dictionary untuk menyimpan nilai fitur kategorikal yang dimasukkan pengguna
fitur_values = {}

# Menggunakan perulangan untuk membuat widget input untuk setiap fitur kategorikal
for fitur in fitur_kategorikal:
    fitur_values[fitur] = st.radio(f'{fitur.capitalize()} - {arti_fitur[fitur]}', ['Iya', 'Tidak'])

# Fitur Numerik
legs = st.number_input('Jumlah Kaki', min_value=0, max_value=8, step=1)

# Tombol untuk Prediksi
if st.button('Prediksi'):
    # Ubah fitur-fitur menjadi format yang dapat digunakan oleh model
    input_features_categorical = np.array([[fitur_values[fitur] == 'Iya' for fitur in fitur_kategorikal]])
    input_features_numerical = np.array([[legs]])

    # Gabungkan kedua array
    input_features = np.concatenate((input_features_categorical, input_features_numerical), axis=1)

    # Lakukan prediksi
    prediction = model.predict(input_features)

    # Tampilkan hasil prediksi di bawah tombol
    tipe_hewan = f'Hewan termasuk ke dalam tipe {prediction[0]} ({tipe_to_nama[prediction[0]]})'
    st.subheader('Hasil Prediksi:')
    st.write(f'Inputan "{nama_hewan}" {tipe_hewan}.')

# Informasi Tambahan
st.info('Aplikasi ini menggunakan model klasifikasi untuk memprediksi kelas hewan berdasarkan fitur-fitur yang dimasukkan.')
