# modul_prediksi_22.py
# ============================================================
# Modul Prediksi ML untuk Dashboard Evaluasi Kompetensi Pegawai v2.2
# ============================================================
#
# Fungsi utama modul ini:
# 1. Memprediksi Unit_kompetensi dan Level_kompetensi yang masih kosong.
# 2. Menggunakan sheet ref sebagai sumber utama Machine Learning.
# 3. Menggunakan gabungan Nama Pelatihan + Silabus sebagai teks referensi.
# 4. Menggunakan gabungan Nama_pelatihan + Silabus sebagai teks query histori.
# 5. Menjaga data asli: kolom yang sudah terisi tidak ditimpa.
# 6. Mendukung SBERT dan TF-IDF Cosine Similarity.
#
# Catatan untuk pemula:
# Modul ini tidak mengevaluasi pegawai kompeten/tidak kompeten.
# Modul ini hanya membantu melengkapi Unit_kompetensi dan Level_kompetensi
# pada sheet Histori Pelatihan.

from __future__ import annotations

import functools
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

from modul_excel_22 import find_col, clean_text, is_empty_value


# ============================================================
# Import library ML dengan fallback aman
# ============================================================

try:
    from sentence_transformers import SentenceTransformer
    SBERT_TERSEDIA = True
except Exception:
    SentenceTransformer = None
    SBERT_TERSEDIA = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_TERSEDIA = True
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None
    SKLEARN_TERSEDIA = False


SBERT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

STATUS_SUDAH_TERISI = "Sudah terisi"
STATUS_CONF_TINGGI = "Diisi otomatis - Confidence tinggi"
STATUS_REVIEW = "Perlu review - belum diisi"
STATUS_RENDAH = "Tidak diisi - Confidence rendah"
STATUS_NAMA_KOSONG = "Nama pelatihan kosong"
STATUS_TIDAK_DIPREDIKSI = "Tidak diprediksi"

THRESHOLD_REVIEW_DEFAULT = 0.60


# ============================================================
# Helper model dan similarity
# ============================================================

@functools.lru_cache(maxsize=1)
def load_sbert_model():
    """Load model SBERT sekali saja.

    lru_cache membuat model tidak di-load berulang-ulang selama sesi Python sama.
    Ini membantu performa saat aplikasi dipakai berulang.
    """
    if not SBERT_TERSEDIA or SentenceTransformer is None:
        raise RuntimeError("sentence-transformers belum tersedia.")
    return SentenceTransformer(SBERT_MODEL_NAME)


def _cosine_matrix_sbert(query_text: List[str], ref_text: List[str]) -> np.ndarray:
    """Membuat matriks cosine similarity dengan SBERT."""
    model = load_sbert_model()
    ref_emb = model.encode(ref_text, normalize_embeddings=True, show_progress_bar=False)
    query_emb = model.encode(query_text, normalize_embeddings=True, show_progress_bar=False)
    return np.matmul(query_emb, ref_emb.T)


def _cosine_matrix_tfidf(query_text: List[str], ref_text: List[str]) -> np.ndarray:
    """Membuat matriks cosine similarity dengan TF-IDF.

    analyzer='char_wb' dan ngram 3-5 cukup cocok untuk variasi ejaan pendek,
    typo ringan, dan nama pelatihan yang mirip.
    """
    if not SKLEARN_TERSEDIA or TfidfVectorizer is None or cosine_similarity is None:
        raise RuntimeError("scikit-learn belum tersedia.")

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        lowercase=True,
    )
    ref_matrix = vectorizer.fit_transform(ref_text)
    query_matrix = vectorizer.transform(query_text)
    return cosine_similarity(query_matrix, ref_matrix)


def build_similarity_matrix(
    query_text: List[str],
    ref_text: List[str],
    metode_prediksi: str = "SBERT",
) -> Tuple[np.ndarray, str]:
    """Membangun matriks similarity sesuai metode pilihan user.

    Jika user memilih SBERT tetapi model gagal, otomatis fallback ke TF-IDF.
    Jika user memilih TF-IDF, langsung pakai TF-IDF.
    """
    metode = str(metode_prediksi or "SBERT").upper().strip()

    if metode == "TF-IDF":
        return _cosine_matrix_tfidf(query_text, ref_text), "TF-IDF"

    # Default: coba SBERT dulu.
    if metode == "SBERT":
        try:
            return _cosine_matrix_sbert(query_text, ref_text), "SBERT"
        except Exception:
            # Fallback agar aplikasi tetap jalan.
            return _cosine_matrix_tfidf(query_text, ref_text), "TF-IDF fallback"

    # Jika input metode aneh, tetap aman.
    try:
        return _cosine_matrix_sbert(query_text, ref_text), "SBERT"
    except Exception:
        return _cosine_matrix_tfidf(query_text, ref_text), "TF-IDF fallback"


# ============================================================
# Membangun referensi ML
# ============================================================

def _gabung_teks(*values) -> str:
    """Menggabungkan beberapa teks menjadi satu teks bersih."""
    parts = []
    for value in values:
        if not is_empty_value(value):
            parts.append(str(value).strip())
    return " ".join(parts).strip()


def reference_from_ref_sheet(df_ref: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Membuat referensi ML dari sheet ref.

    Sheet ref adalah sumber utama v2.2.

    Teks referensi:
        Nama Pelatihan + Silabus

    Kolom output:
    - nama
    - silabus
    - teks_ref
    - unit
    - level
    - source
    - prioritas
    """
    if df_ref is None or len(df_ref) == 0:
        return pd.DataFrame(columns=["nama", "silabus", "teks_ref", "unit", "level", "source", "prioritas"])

    nama_col = find_col(
        df_ref,
        ["Nama Pelatihan", "Nama_pelatihan", "Nama_Pelatihan", "Nama Program", "Program"],
        allow_contains=True,
    )
    silabus_col = find_col(
        df_ref,
        ["Silabus", "Syllabus", "Materi Pelatihan", "Deskripsi Pelatihan"],
        allow_contains=True,
    )
    unit_col = find_col(
        df_ref,
        ["Unit Kompetensi", "Unit_Kompetensi", "Unit_kompetensi", "UK"],
        allow_contains=True,
    )
    level_col = find_col(
        df_ref,
        ["Level Kompetensi", "Level_Kompetensi", "Level_kompetensi", "Level"],
        allow_contains=True,
    )

    if nama_col is None or unit_col is None:
        return pd.DataFrame(columns=["nama", "silabus", "teks_ref", "unit", "level", "source", "prioritas"])

    rows = []
    for _, r in df_ref.iterrows():
        nama = r.get(nama_col, "")
        silabus = r.get(silabus_col, "") if silabus_col else ""
        unit = r.get(unit_col, "")
        level = r.get(level_col, "") if level_col else ""

        teks_ref = _gabung_teks(nama, silabus)

        if clean_text(teks_ref) == "" or is_empty_value(unit):
            continue

        rows.append({
            "nama": nama,
            "silabus": silabus,
            "teks_ref": teks_ref,
            "unit": unit,
            "level": level,
            "source": "Sheet ref",
            "prioritas": 1,
        })

    return pd.DataFrame(rows)


def reference_from_labeled_histori(df_histori: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Membuat referensi tambahan dari histori yang sudah punya Unit_kompetensi.

    Teks referensi:
        Nama_pelatihan + Silabus

    Referensi histori dipakai setelah sheet ref.
    """
    if df_histori is None or len(df_histori) == 0:
        return pd.DataFrame(columns=["nama", "silabus", "teks_ref", "unit", "level", "source", "prioritas"])

    nama_col = find_col(
        df_histori,
        ["Nama_pelatihan", "Nama Pelatihan", "Nama_Pelatihan"],
        allow_contains=True,
    )
    silabus_col = find_col(df_histori, ["Silabus"], allow_contains=True)
    unit_col = find_col(
        df_histori,
        ["Unit_kompetensi", "Unit Kompetensi", "Unit_Kompetensi"],
        allow_contains=True,
    )
    level_col = find_col(
        df_histori,
        ["Level_kompetensi", "Level Kompetensi", "Level_Kompetensi"],
        allow_contains=True,
    )

    if nama_col is None or unit_col is None:
        return pd.DataFrame(columns=["nama", "silabus", "teks_ref", "unit", "level", "source", "prioritas"])

    rows = []
    for _, r in df_histori.iterrows():
        nama = r.get(nama_col, "")
        silabus = r.get(silabus_col, "") if silabus_col else ""
        unit = r.get(unit_col, "")
        level = r.get(level_col, "") if level_col else ""

        teks_ref = _gabung_teks(nama, silabus)

        if clean_text(teks_ref) == "" or is_empty_value(unit):
            continue

        rows.append({
            "nama": nama,
            "silabus": silabus,
            "teks_ref": teks_ref,
            "unit": unit,
            "level": level,
            "source": "Histori sudah terisi",
            "prioritas": 2,
        })

    return pd.DataFrame(rows)


def reference_from_standar(df_standar: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Membuat fallback referensi dari standar kompetensi.

    Ini bukan sumber utama prediksi pelatihan.
    Dipakai hanya agar aplikasi tetap punya referensi jika sheet ref tidak lengkap.
    """
    if df_standar is None or len(df_standar) == 0:
        return pd.DataFrame(columns=["nama", "silabus", "teks_ref", "unit", "level", "source", "prioritas"])

    jabatan_col = find_col(df_standar, ["Jabatan"], allow_contains=True)
    unit_col = find_col(df_standar, ["Unit_Kompetensi", "Unit Kompetensi", "Unit_kompetensi"], allow_contains=True)
    level_col = find_col(df_standar, ["Level_Kompetensi", "Level Kompetensi", "Level_kompetensi"], allow_contains=True)
    desc_col = find_col(df_standar, ["Deskripsi_Level", "Deskripsi Level", "Deskripsi"], allow_contains=True)

    if unit_col is None:
        return pd.DataFrame(columns=["nama", "silabus", "teks_ref", "unit", "level", "source", "prioritas"])

    rows = []
    for _, r in df_standar.iterrows():
        unit = r.get(unit_col, "")
        if is_empty_value(unit):
            continue

        jabatan = r.get(jabatan_col, "") if jabatan_col else ""
        level = r.get(level_col, "") if level_col else ""
        desc = r.get(desc_col, "") if desc_col else ""

        teks_ref = _gabung_teks(unit, jabatan, desc)

        rows.append({
            "nama": unit,
            "silabus": desc,
            "teks_ref": teks_ref,
            "unit": unit,
            "level": level,
            "source": "Standar Kompetensi",
            "prioritas": 3,
        })

    return pd.DataFrame(rows)


def build_reference_table(
    df_histori: Optional[pd.DataFrame],
    df_standar: Optional[pd.DataFrame],
    df_ref: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Menggabungkan semua sumber referensi ML dengan prioritas.

    Urutan:
    1. Sheet ref
    2. Histori yang sudah terisi
    3. Standar kompetensi
    """
    ref_sheet = reference_from_ref_sheet(df_ref)
    ref_histori = reference_from_labeled_histori(df_histori)
    ref_standar = reference_from_standar(df_standar)

    ref_parts = []
    if len(ref_sheet) > 0:
        ref_parts.append(ref_sheet)
    if len(ref_histori) > 0:
        ref_parts.append(ref_histori)
    if len(ref_standar) > 0:
        ref_parts.append(ref_standar)

    summary = {
        "jumlah_ref_sheet": int(len(ref_sheet)),
        "jumlah_ref_histori": int(len(ref_histori)),
        "jumlah_ref_standar": int(len(ref_standar)),
    }

    if not ref_parts:
        return pd.DataFrame(columns=["nama", "silabus", "teks_ref", "unit", "level", "source", "prioritas", "_clean"]), summary

    df_ref_all = pd.concat(ref_parts, ignore_index=True)
    df_ref_all["_clean"] = df_ref_all["teks_ref"].apply(clean_text)
    df_ref_all["_clean_nama"] = df_ref_all["nama"].apply(clean_text)

    df_ref_all = df_ref_all[
        (df_ref_all["_clean"] != "")
        & (~df_ref_all["unit"].apply(is_empty_value))
    ].copy()

    # Prioritas kecil lebih utama. Jika referensi duplikat, ref akan dipertahankan.
    df_ref_all = df_ref_all.sort_values(["prioritas"]).drop_duplicates(
        subset=["_clean", "unit"],
        keep="first",
    ).reset_index(drop=True)

    return df_ref_all, summary


# ============================================================
# Keyword rule ringan
# ============================================================

def keyword_rule_predict(text: str, df_standar: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
    """Rule sederhana untuk membantu jika similarity rendah.

    Rule ini sengaja ringan dan tidak terlalu agresif.
    Dipakai hanya untuk membantu kandidat ketika skor ML berada di area review.
    """
    text_clean = clean_text(text)
    if text_clean == "":
        return None

    rules = [
        (
            [
                "data analytics", "data analitik", "analitik", "analytics",
                "statistika", "statistik", "olah data", "pengolahan data",
                "excel", "spreadsheet", "visualisasi", "dashboard",
                "power bi", "tableau", "python", "sql",
            ],
            ["Pengolahan Data", "Penyajian Data Visual"],
        ),
        (
            [
                "informasi", "knowledge management", "manajemen data",
                "sistem informasi", "database", "taxpayer account", "coretax",
            ],
            ["Manajemen Informasi", "Manjemen Informasi"],
        ),
        (
            [
                "aplikasi perkantoran", "office", "word", "powerpoint",
                "komputer", "digital literacy", "literasi digital",
            ],
            ["Penguasaan Aplikasi Perkantoran"],
        ),
        (
            ["dokumen", "arsip", "kearsipan", "surat", "naskah dinas"],
            ["Manajemen Dokumen", "Penyusunan Laporan Teknis"],
        ),
        (
            ["laporan", "report", "telaah", "notula"],
            ["Penyusunan Laporan Teknis"],
        ),
        (
            ["kebijakan", "policy", "analisis kebijakan"],
            ["Analisis Kebijakan"],
        ),
        (
            [
                "regulasi", "peraturan", "undang", "hukum",
                "legal drafting", "disiplin pegawai",
            ],
            ["Regulasi Peraturan Perundang-Undangan"],
        ),
        (
            ["pelayanan", "layanan publik", "komunikasi", "public speaking", "service"],
            ["Pelayanan Administratif"],
        ),
    ]

    unit_col = find_col(df_standar, ["Unit_Kompetensi", "Unit Kompetensi", "Unit_kompetensi"], allow_contains=True) if df_standar is not None else None
    level_col = find_col(df_standar, ["Level_Kompetensi", "Level Kompetensi", "Level_kompetensi"], allow_contains=True) if df_standar is not None else None

    available = []
    if df_standar is not None and unit_col:
        for _, r in df_standar.iterrows():
            available.append({
                "unit": r.get(unit_col, ""),
                "level": r.get(level_col, "") if level_col else "",
            })

    for keywords, target_units in rules:
        if any(k in text_clean for k in keywords):
            # Utamakan unit yang memang ada di standar kompetensi.
            for target in target_units:
                for item in available:
                    if clean_text(item["unit"]) == clean_text(target):
                        return item

            # Jika tidak ditemukan di standar, tetap kembalikan target pertama.
            return {"unit": target_units[0], "level": ""}

    return None


# ============================================================
# Fungsi utama prediksi
# ============================================================

def prediksi_histori_unit_level(
    df_histori: pd.DataFrame,
    df_standar: pd.DataFrame,
    df_ref: Optional[pd.DataFrame] = None,
    batas_confidence: int = 75,
    metode_prediksi: str = "SBERT",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Memprediksi Unit_kompetensi dan Level_kompetensi pada histori.

    Aturan utama:
    - Prediksi hanya mengisi kolom yang kosong.
    - Data asli yang sudah terisi tidak ditimpa.
    - Sheet ref menjadi sumber utama.
    - Teks query histori = Nama_pelatihan + Silabus.
    - Teks referensi ref = Nama Pelatihan + Silabus.
    """
    hasil = df_histori.copy()

    nama_col = find_col(
        hasil,
        ["Nama_pelatihan", "Nama Pelatihan", "Nama_Pelatihan"],
        allow_contains=True,
    )
    silabus_col = find_col(hasil, ["Silabus"], allow_contains=True)
    unit_col = find_col(
        hasil,
        ["Unit_kompetensi", "Unit Kompetensi", "Unit_Kompetensi"],
        allow_contains=True,
    )
    level_col = find_col(
        hasil,
        ["Level_kompetensi", "Level Kompetensi", "Level_Kompetensi"],
        allow_contains=True,
    )

    if nama_col is None:
        hasil["Status_Prediksi"] = "Kolom Nama_pelatihan tidak ditemukan"
        return hasil, {
            "pesan": "Prediksi tidak berjalan karena kolom Nama_pelatihan tidak ditemukan.",
            "metode_prediksi": "Tidak berjalan",
            "jumlah_unit_diisi": 0,
            "jumlah_level_diisi": 0,
            "jumlah_perlu_review": 0,
            "jumlah_confidence_rendah": 0,
        }

    # Jika kolom target belum ada, buat kolomnya.
    if unit_col is None:
        hasil["Unit_kompetensi"] = ""
        unit_col = "Unit_kompetensi"

    if level_col is None:
        hasil["Level_kompetensi"] = ""
        level_col = "Level_kompetensi"

    if silabus_col is None:
        hasil["Silabus"] = ""
        silabus_col = "Silabus"

    unit_kosong_awal = int(hasil[unit_col].apply(is_empty_value).sum())
    level_kosong_awal = int(hasil[level_col].apply(is_empty_value).sum())

    # Kolom audit prediksi.
    hasil["Unit_Asli"] = hasil[unit_col]
    hasil["Level_Asli"] = hasil[level_col]
    hasil["Prediksi_Unit_Kompetensi"] = hasil[unit_col]
    hasil["Prediksi_Level_Kompetensi"] = hasil[level_col]
    hasil["Nama_Pelatihan_Termirip"] = ""
    hasil["Silabus_Termirip"] = ""
    hasil["Confidence_Prediksi"] = 0.0
    hasil["Sumber_Prediksi"] = ""
    hasil["Metode_Prediksi"] = ""
    hasil["Status_Prediksi"] = STATUS_TIDAK_DIPREDIKSI

    df_ref_all, ref_summary = build_reference_table(
        df_histori=hasil,
        df_standar=df_standar,
        df_ref=df_ref,
    )

    if len(df_ref_all) == 0:
        return hasil, {
            "pesan": "Prediksi tidak berjalan karena tidak ada referensi Unit Kompetensi.",
            "metode_prediksi": "Tidak berjalan",
            "jumlah_unit_diisi": 0,
            "jumlah_level_diisi": 0,
            "unit_kosong_awal": unit_kosong_awal,
            "level_kosong_awal": level_kosong_awal,
            "unit_kosong_akhir": unit_kosong_awal,
            "level_kosong_akhir": level_kosong_awal,
            **ref_summary,
        }

    threshold_isi = float(batas_confidence) / 100.0
    threshold_review = THRESHOLD_REVIEW_DEFAULT

    jumlah_unit_diisi = 0
    jumlah_level_diisi = 0

    # Query histori: Nama_pelatihan + Silabus
    query_text_raw = (
        hasil[nama_col].fillna("").astype(str)
        + " "
        + hasil[silabus_col].fillna("").astype(str)
    )
    query_text = query_text_raw.apply(clean_text).tolist()

    ref_text = df_ref_all["_clean"].tolist()

    similarities, metode_dipakai = build_similarity_matrix(
        query_text=query_text,
        ref_text=ref_text,
        metode_prediksi=metode_prediksi,
    )

    for i in range(len(hasil)):
        nama_asli = hasil.at[i, nama_col]
        silabus_asli = hasil.at[i, silabus_col]
        teks_query_bersih = query_text[i]

        unit_sudah_isi = not is_empty_value(hasil.at[i, unit_col])
        level_sudah_isi = not is_empty_value(hasil.at[i, level_col])

        hasil.at[i, "Metode_Prediksi"] = metode_dipakai

        if clean_text(nama_asli) == "":
            hasil.at[i, "Status_Prediksi"] = STATUS_NAMA_KOSONG
            continue

        if unit_sudah_isi and level_sudah_isi:
            hasil.at[i, "Status_Prediksi"] = STATUS_SUDAH_TERISI
            hasil.at[i, "Confidence_Prediksi"] = 100.0
            hasil.at[i, "Sumber_Prediksi"] = "Data asli"
            continue

        sim_row = similarities[i]
        best_idx = int(np.argmax(sim_row))
        best_score = float(sim_row[best_idx])

        ref = df_ref_all.iloc[best_idx]
        best_unit = ref.get("unit", "")
        best_level = ref.get("level", "")
        best_source = ref.get("source", "")
        best_name = ref.get("nama", "")
        best_silabus = ref.get("silabus", "")

        # Exact match: nama pelatihan saja.
        # Ini penting agar pelatihan dengan nama sama langsung cocok walau silabus sedikit berbeda.
        q_nama_only = clean_text(nama_asli)
        exact_match_nama = df_ref_all[df_ref_all["_clean_nama"] == q_nama_only]

        if len(exact_match_nama) > 0:
            ref = exact_match_nama.iloc[0]
            best_score = 1.0
            best_unit = ref.get("unit", "")
            best_level = ref.get("level", "")
            best_source = str(ref.get("source", "")) + " - Exact Match Nama"
            best_name = ref.get("nama", "")
            best_silabus = ref.get("silabus", "")

        # Exact match: nama + silabus.
        q_full = teks_query_bersih
        exact_match_full = df_ref_all[df_ref_all["_clean"] == q_full]
        if len(exact_match_full) > 0:
            ref = exact_match_full.iloc[0]
            best_score = 1.0
            best_unit = ref.get("unit", "")
            best_level = ref.get("level", "")
            best_source = str(ref.get("source", "")) + " - Exact Match Full"
            best_name = ref.get("nama", "")
            best_silabus = ref.get("silabus", "")

        # Keyword rule ringan jika skor masih belum cukup.
        rule = keyword_rule_predict(_gabung_teks(nama_asli, silabus_asli), df_standar)
        if rule is not None and best_score < threshold_review:
            best_unit = rule.get("unit", "")
            best_level = rule.get("level", "")
            best_source = "Keyword Rule"
            best_score = max(best_score, 0.62)
            best_name = best_unit
            best_silabus = ""

        hasil.at[i, "Nama_Pelatihan_Termirip"] = best_name
        hasil.at[i, "Silabus_Termirip"] = best_silabus
        hasil.at[i, "Confidence_Prediksi"] = round(best_score * 100, 2)
        hasil.at[i, "Sumber_Prediksi"] = best_source

        # Sumber aman untuk isi otomatis.
        # Standar Kompetensi sebagai fallback dibuat lebih hati-hati:
        # boleh isi otomatis jika exact match atau score sangat tinggi.
        sumber_aman = (
            "Sheet ref" in str(best_source)
            or "Histori" in str(best_source)
            or "Exact Match" in str(best_source)
            or "Keyword Rule" in str(best_source)
        )

        sumber_standar_saja = "Standar Kompetensi" in str(best_source) and not sumber_aman
        if sumber_standar_saja and best_score < 0.90:
            sumber_aman = False

        if best_score >= threshold_isi and sumber_aman and not is_empty_value(best_unit):
            if not unit_sudah_isi:
                hasil.at[i, "Prediksi_Unit_Kompetensi"] = best_unit
                hasil.at[i, unit_col] = best_unit
                jumlah_unit_diisi += 1

            if not level_sudah_isi and not is_empty_value(best_level):
                hasil.at[i, "Prediksi_Level_Kompetensi"] = best_level
                hasil.at[i, level_col] = best_level
                jumlah_level_diisi += 1

            hasil.at[i, "Status_Prediksi"] = STATUS_CONF_TINGGI

        elif best_score >= threshold_review and not is_empty_value(best_unit):
            hasil.at[i, "Status_Prediksi"] = STATUS_REVIEW

        else:
            hasil.at[i, "Status_Prediksi"] = STATUS_RENDAH

    unit_kosong_akhir = int(hasil[unit_col].apply(is_empty_value).sum())
    level_kosong_akhir = int(hasil[level_col].apply(is_empty_value).sum())

    jumlah_tinggi = int((hasil["Status_Prediksi"] == STATUS_CONF_TINGGI).sum())
    jumlah_review = int((hasil["Status_Prediksi"] == STATUS_REVIEW).sum())
    jumlah_rendah = int((hasil["Status_Prediksi"] == STATUS_RENDAH).sum())
    jumlah_sudah = int((hasil["Status_Prediksi"] == STATUS_SUDAH_TERISI).sum())

    pesan = (
        f"Prediksi {metode_dipakai} selesai. "
        f"Unit diisi: {jumlah_unit_diisi}. "
        f"Level diisi: {jumlah_level_diisi}. "
        f"Perlu review: {jumlah_review}. "
        f"Confidence rendah/tidak diisi: {jumlah_rendah}."
    )

    meta = {
        "pesan": pesan,
        "metode_prediksi": metode_dipakai,
        "jumlah_unit_diisi": jumlah_unit_diisi,
        "jumlah_level_diisi": jumlah_level_diisi,
        "unit_kosong_awal": unit_kosong_awal,
        "level_kosong_awal": level_kosong_awal,
        "unit_kosong_akhir": unit_kosong_akhir,
        "level_kosong_akhir": level_kosong_akhir,
        "jumlah_confidence_tinggi": jumlah_tinggi,
        "jumlah_perlu_review": jumlah_review,
        "jumlah_confidence_rendah": jumlah_rendah,
        "jumlah_sudah_terisi": jumlah_sudah,
        "batas_confidence": batas_confidence,
        "threshold_review": int(threshold_review * 100),
        **ref_summary,
    }

    return hasil, meta


# Alias kompatibilitas jika ada kode lama yang masih memanggil nama ini.
prediksi_histori_unit_level_sbert = prediksi_histori_unit_level
prediksi_histori_unit_level_tfidf = prediksi_histori_unit_level
