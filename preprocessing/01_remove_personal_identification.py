import pandas as pd
import requests
import json
import time

# =========================
# Konfigurasi
# =========================
INPUT_PATH = r"C:\Users\Yohanes\Documents\Lab\Preprocessing\dataset_anonim_phase3_INPUT_Ollama.xlsx"
OUTPUT_PATH = r"C:\Users\Yohanes\Documents\Lab\Preprocessing\dataset_anonim_phase3_FINAL_Ollama.xlsx"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma4:e4b" #"gemma3:27b" #deepseek-r1:8b, qwen3:30b, gemma3:12b, gemma3:27b

# Prompt dasar / perintah
PERINTAH = (
"Kamu adalah ahli tata bahasa, baik bahasa Indonesia maupun Inggris. "
"Tugasmu adalah mengecek apakah teks yang akan saya berikan ini mengandung informasi personal tertentu yang spesifik. "
"Informasi personal seperti nama orang, nama organisasi, nama sekolah, nama instansi, nama universitas atau nama perguruan tinggi, atau nama kelembagaan nasional atau internasional. "
"Jika teks mengandung informasi personal tersebut, maka hilangkan informasi tersebut sedemikian rupa sehingga struktur kalimat utuh masih bermakna. "
"Jangan hilangkan informasi nama lembaga yang bernama BPS atau Badan Pusat Statistik, informasi ini penting untuk tetap ada. "
"Saya hanya ingin menghilangkan informasi data personal, namun masih tetap menjaga semantik kalimat secara utuh, sehingga meskipun dihilangkan, kalimat masih enak untuk dibaca. "
"Jangan mengganti secara masif kata per kata atau struktur kalimat awal setelah informasi personal dihilangkan. "
"Responmu singkat: berikan kalimat yang sudah dibersihkan dari informasi personal, tanpa penjelasan apapun. Respon tanpa salam pembuka, tanpa penegasan perintah. "
"Lalu respon berupa kalimat text flat tanpa newline maupun ganti baris, tanpa bulleting,tanpa petik dua (jika ada petik dua ganti ke petik satu). "
"Jangan menghilangkan informasi referensi data yang diminta, misal susenas, podes, sakernas, dan label nama yang menunjuk sebuah entitas data/dataset/dokumen. "
"Untuk kalimat awalan yang konteksnya sanjungan, salam sapaan yang tidak ada hubungannya dengan data request, hilangkan. "
"Untuk kalimat bagian akhir yang terdeteksi seperti kalimat salam penutup, kalimat terima kasih, atau salam penutup lainnya, hilangkan. "
"Serta sebisa mungkin langsung konstruk kalimat tanya yang sesuai. Berikut kalimat awal yang dianggap mengandung informasi personal: "
" "
)

# =========================
# Load file XLSX
# =========================
USERDATA = pd.read_excel(INPUT_PATH)

# Pastikan kolom paraphrase ada
if "paraphrase" not in USERDATA.columns:
    USERDATA["paraphrase"] = ""

# =========================
# Looping tiap row
# =========================
for idx, row in USERDATA.iterrows():
    SENTENCE = str(row["question"])  # ambil kolom question
    payload = {
        "model": MODEL_NAME,
        "prompt": PERINTAH + SENTENCE
    }

    try:
        # Request ke Ollama local API
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=1200)

        paraphrase_text = ""
        # Response streaming
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    paraphrase_text += data["response"]

        # Masukkan hasil ke kolom paraphrase di baris yang sesuai
        USERDATA.at[idx, "paraphrase"] = paraphrase_text.strip()

        # Optional: print progress
        print(f"[{idx+1}/{len(USERDATA)}] Done")

        # Delay sebentar supaya tidak terlalu agresif (opsional)
        time.sleep(0.5)

    except requests.exceptions.RequestException as e:
        print(f"Error at row {idx}: {e}")
        USERDATA.at[idx, "paraphrase"] = "ERROR"

# =========================
# Simpan kembali ke XLSX
# =========================
USERDATA.to_excel(OUTPUT_PATH, index=False)
print(f"Done! File tersimpan di: {OUTPUT_PATH}")