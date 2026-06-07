import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def run_preprocessing():
    print("Memulai otomatisasi preprocessing...")
    
    # Mendapatkan direktori dari skrip ini berada (folder preprocessing)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Menggabungkan path ke file CSV yang ada di satu tingkat di atasnya
    file_path = os.path.join(current_dir, '..', 'heart_failure_clinical_records.csv')
    df = pd.read_csv(file_path)
    print(f"Shape data awal: {df.shape}")
    
    # 2. Membuang kolom 'time' untuk mencegah data leakage
    df_clean = df.drop(columns=['time'])
    
    # 3. Menghapus data duplikat
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"Jumlah baris duplikat yang dihapus: {initial_rows - len(df_clean)}")
    
    # 4. Standarisasi fitur numerik
    numeric_features = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']
    scaler = StandardScaler()
    df_clean[numeric_features] = scaler.fit_transform(df_clean[numeric_features])
    print("Fitur numerik berhasil distandarisasi.")
    
    # 5. Menyimpan data bersih menjadi file baru
    output_file = 'heart_failure_preprocessing.csv'
    df_clean.to_csv(output_file, index=False)
    print(f"Preprocessing selesai! Data bersih disimpan sebagai: {output_file}")
    
    # Mengembalikan data yang siap dilatih sesuai instruksi Dicoding
    return df_clean

# Blok ini memastikan fungsi run_preprocessing() berjalan otomatis saat file dieksekusi
if __name__ == "__main__":
    run_preprocessing()