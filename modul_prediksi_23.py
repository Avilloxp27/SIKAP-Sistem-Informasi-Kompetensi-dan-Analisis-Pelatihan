# modul_prediksi_23.py
# ============================================================
# Modul Prediksi ML untuk Dashboard Evaluasi Kompetensi Pegawai v2.3
# ============================================================
#
# PERUBAHAN DARI v2.2:
# [OPT-1] Vektorisasi loop prediksi
#         Loop for i in range(len(hasil)) diganti operasi kolom NumPy/Pandas.
#         Tidak ada lagi hasil.at[i, kolom] di dalam loop utama.
#         Semua assign dilakukan sekaligus dengan np.where / pd.Series.
#
# [OPT-2] Cache embedding SBERT di Streamlit session_state
#         Embedding referensi disimpan di st.session_state berdasarkan
#         hash teks referensi. Selama referensi tidak berubah, embedding
#         tidak dihitung ulang walau user mengubah parameter lain.
#
# [OPT-3] Batch encode untuk data > 1000 baris
#         Query encoding dilakukan per batch (default 512 baris).
#         Mencegah OOM dan membuat progress bar Streamlit bisa tampil.
#
# Kompatibilitas:
# - Drop-in replacement untuk modul_prediksi_22.py.
# - Nama fungsi publik identik: prediksi_histori_unit_level().
# - Alias lama tetap tersedia.
# - Tidak membutuhkan perubahan di modul_evaluasi_22.py maupun app_2.2.py.
#   Cukup ganti baris import di modul_evaluasi_22.py:
#   LAMA: from modul_prediksi_22 import prediksi_histori_unit_level
#   BARU: from modul_prediksi_23 import prediksi_histori_unit_level

from __future__ import annotations

import functools
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from modul_excel_22 import clean_text, find_col, is_empty_value

logger = logging.getLogger(__name__)


# ============================================================
# Import library ML dengan fallback aman
# ============================================================

try:
    from sentence_transformers import SentenceTransformer
    SBERT_TERSEDIA = True
except Exception:
    SentenceTransformer = None  # type: ignore
    SBERT_TERSEDIA = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_TERSEDIA = True
except Exception:
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore
    SKLEARN_TERSEDIA = False

# Streamlit session_state tersedia saat dijalankan lewat Streamlit.
# Di luar Streamlit (unit test, script CLI), fallback ke dict biasa.
try:
    import streamlit as st
    _STREAMLIT_TERSEDIA = True
except Exception:
    st = None  # type: ignore
    _STREAMLIT_TERSEDIA = False


# ============================================================
# Konstanta
# ============================================================

SBERT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

STATUS_SUDAH_TERISI  = "Sudah terisi"
STATUS_CONF_TINGGI   = "Diisi otomatis - Confidence tinggi"
STATUS_REVIEW        = "Perlu review - belum diisi"
STATUS_RENDAH        = "Tidak diisi - Confidence rendah"
STATUS_NAMA_KOSONG   = "Nama pelatihan kosong"
STATUS_TIDAK_DIPREDIKSI = "Tidak diprediksi"

THRESHOLD_REVIEW_DEFAULT = 0.60

# [OPT-3] Ukuran batch default untuk encode.
# Naikkan jika RAM cukup, turunkan jika sering OOM.
BATCH_SIZE_DEFAULT = 512

# Key yang dipakai di st.session_state untuk menyimpan cache embedding.
_CACHE_KEY_REF_EMB  = "_prediksi23_ref_emb"
_CACHE_KEY_REF_HASH = "_prediksi23_ref_hash"

# Fallback dict jika dijalankan di luar Streamlit.
_local_cache: Dict[str, Any] = {}


# ============================================================
# [OPT-1] Helper cache session state
# ============================================================

def _session_get(key: str) -> Any:
    """Ambil nilai dari session_state (Streamlit) atau dict lokal."""
    if _STREAMLIT_TERSEDIA and st is not None:
        return st.session_state.get(key)
    return _local_cache.get(key)


def _session_set(key: str, value: Any) -> None:
    """Simpan nilai ke session_state (Streamlit) atau dict lokal."""
    if _STREAMLIT_TERSEDIA and st is not None:
        st.session_state[key] = value
    else:
        _local_cache[key] = value


def _hash_teks_list(teks_list: List[str]) -> str:
    """Buat hash MD5 ringkas dari list teks referensi.

    Dipakai sebagai cache key untuk mendeteksi apakah referensi berubah.
    """
    gabung = "\n".join(teks_list)
    return hashlib.md5(gabung.encode("utf-8", errors="replace")).hexdigest()


# ============================================================
# Load model SBERT
# ============================================================

@functools.lru_cache(maxsize=1)
def load_sbert_model() -> "SentenceTransformer":
    """Load model SBERT sekali saja selama proses Python hidup.

    lru_cache menjamin model tidak di-load ulang walau fungsi dipanggil
    berkali-kali. Ini berbeda dari cache embedding — model di-cache di
    level proses, embedding di-cache di level session Streamlit.
    """
    if not SBERT_TERSEDIA or SentenceTransformer is None:
        raise RuntimeError("sentence-transformers belum tersedia.")
    return SentenceTransformer(SBERT_MODEL_NAME)


# ============================================================
# [OPT-3] Batch encode
# ============================================================

def _encode_batch(
    model: "SentenceTransformer",
    teks_list: List[str],
    batch_size: int = BATCH_SIZE_DEFAULT,
) -> np.ndarray:
    """Encode teks dalam batch untuk menghindari OOM pada data besar.

    Untuk data <= batch_size, hasilnya sama persis dengan encode biasa.
    Untuk data > batch_size, dilakukan chunk dan digabung kembali.

    Parameter:
        model      : model SBERT yang sudah di-load.
        teks_list  : list teks yang akan di-encode.
        batch_size : jumlah teks per batch (default 512).

    Return:
        np.ndarray shape (N, dim), normalized.
    """
    if len(teks_list) == 0:
        return np.array([])

    if len(teks_list) <= batch_size:
        return model.encode(
            teks_list,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    chunks = []
    for start in range(0, len(teks_list), batch_size):
        chunk = teks_list[start : start + batch_size]
        emb = model.encode(
            chunk,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        chunks.append(emb)
        logger.debug("Batch encode: %d/%d selesai.", start + len(chunk), len(teks_list))

    return np.vstack(chunks)


# ============================================================
# [OPT-2] Encode referensi dengan cache session state
# ============================================================

def _encode_ref_dengan_cache(
    model: "SentenceTransformer",
    ref_text: List[str],
    batch_size: int = BATCH_SIZE_DEFAULT,
) -> np.ndarray:
    """Encode teks referensi dengan cache di session_state.

    Jika teks referensi tidak berubah sejak run sebelumnya, embedding
    langsung diambil dari cache tanpa encode ulang.

    Ini sangat berguna saat user mengubah parameter (confidence, metode)
    tapi file Excel yang diupload sama — referensinya identik.
    """
    ref_hash = _hash_teks_list(ref_text)

    cached_hash = _session_get(_CACHE_KEY_REF_HASH)
    cached_emb  = _session_get(_CACHE_KEY_REF_EMB)

    if cached_hash == ref_hash and cached_emb is not None:
        logger.debug("Cache hit: embedding referensi dipakai dari session_state.")
        return cached_emb

    logger.debug("Cache miss: encode ulang %d teks referensi.", len(ref_text))
    ref_emb = _encode_batch(model, ref_text, batch_size=batch_size)

    _session_set(_CACHE_KEY_REF_HASH, ref_hash)
    _session_set(_CACHE_KEY_REF_EMB, ref_emb)

    return ref_emb


# ============================================================
# Similarity matrix
# ============================================================

def _cosine_matrix_sbert(
    query_text: List[str],
    ref_text: List[str],
    batch_size: int = BATCH_SIZE_DEFAULT,
) -> np.ndarray:
    """Cosine similarity dengan SBERT + cache referensi + batch encode query."""
    model = load_sbert_model()

    # [OPT-2] Referensi di-cache, tidak encode ulang jika sama.
    ref_emb = _encode_ref_dengan_cache(model, ref_text, batch_size=batch_size)

    # [OPT-3] Query di-encode per batch.
    query_emb = _encode_batch(model, query_text, batch_size=batch_size)

    return np.matmul(query_emb, ref_emb.T)


def _cosine_matrix_tfidf(
    query_text: List[str],
    ref_text: List[str],
) -> np.ndarray:
    """Cosine similarity dengan TF-IDF char ngram (3-5).

    TF-IDF cukup cepat secara native dan tidak perlu cache khusus.
    """
    if not SKLEARN_TERSEDIA or TfidfVectorizer is None or cosine_similarity is None:
        raise RuntimeError("scikit-learn belum tersedia.")

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        lowercase=True,
    )
    ref_matrix   = vectorizer.fit_transform(ref_text)
    query_matrix = vectorizer.transform(query_text)
    return cosine_similarity(query_matrix, ref_matrix)


def build_similarity_matrix(
    query_text: List[str],
    ref_text: List[str],
    metode_prediksi: str = "SBERT",
    batch_size: int = BATCH_SIZE_DEFAULT,
) -> Tuple[np.ndarray, str]:
    """Bangun matriks similarity sesuai metode dengan fallback otomatis."""
    metode = str(metode_prediksi or "SBERT").upper().strip()

    if metode == "TF-IDF":
        return _cosine_matrix_tfidf(query_text, ref_text), "TF-IDF"

    try:
        return _cosine_matrix_sbert(query_text, ref_text, batch_size=batch_size), "SBERT"
    except Exception as exc:
        logger.warning("SBERT gagal (%s), fallback ke TF-IDF.", exc)
        return _cosine_matrix_tfidf(query_text, ref_text), "TF-IDF fallback"


# ============================================================
# Helper teks
# ============================================================

def _gabung_teks(*values: Any) -> str:
    parts = [str(v).strip() for v in values if not is_empty_value(v)]
    return " ".join(parts).strip()


# ============================================================
# Bangun tabel referensi ML
# ============================================================

def _build_ref_dari_df(
    df: Optional[pd.DataFrame],
    nama_candidates: List[str],
    silabus_candidates: List[str],
    unit_candidates: List[str],
    level_candidates: List[str],
    source_label: str,
    prioritas: int,
    gabung_kolom_extra: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Helper umum untuk membuat tabel referensi dari satu DataFrame.

    Menggantikan tiga fungsi terpisah reference_from_ref_sheet,
    reference_from_labeled_histori, reference_from_standar dengan satu
    fungsi yang lebih efisien (memakai operasi kolom, bukan iterrows).
    """
    COLS = ["nama", "silabus", "teks_ref", "unit", "level", "source", "prioritas"]

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=COLS)

    nama_col    = find_col(df, nama_candidates, allow_contains=True)
    silabus_col = find_col(df, silabus_candidates, allow_contains=True)
    unit_col    = find_col(df, unit_candidates, allow_contains=True)
    level_col   = find_col(df, level_candidates, allow_contains=True)

    if nama_col is None or unit_col is None:
        return pd.DataFrame(columns=COLS)

    # Operasi kolom — tidak ada iterrows.
    out = pd.DataFrame()
    out["nama"]    = df[nama_col].fillna("").astype(str).str.strip()
    out["silabus"] = df[silabus_col].fillna("").astype(str).str.strip() if silabus_col else ""
    out["unit"]    = df[unit_col].fillna("").astype(str).str.strip()
    out["level"]   = df[level_col].fillna("").astype(str).str.strip() if level_col else ""

    # Gabungkan kolom tambahan ke teks_ref jika ada (misal Deskripsi untuk standar).
    teks_parts = [out["nama"], out["silabus"]]
    if gabung_kolom_extra:
        for extra_candidates in gabung_kolom_extra:
            col = find_col(df, extra_candidates, allow_contains=True)
            if col:
                teks_parts.append(df[col].fillna("").astype(str).str.strip())

    out["teks_ref"] = teks_parts[0]
    for part in teks_parts[1:]:
        mask = part.str.strip() != ""
        out.loc[mask, "teks_ref"] = out.loc[mask, "teks_ref"] + " " + part[mask]
    out["teks_ref"] = out["teks_ref"].str.strip()

    out["source"]   = source_label
    out["prioritas"] = prioritas

    # Filter baris yang tidak valid.
    valid = (
        (out["teks_ref"].str.strip() != "")
        & (~out["unit"].apply(is_empty_value))
    )
    return out[valid][COLS].reset_index(drop=True)


def reference_from_ref_sheet(df_ref: Optional[pd.DataFrame]) -> pd.DataFrame:
    return _build_ref_dari_df(
        df_ref,
        nama_candidates=["Nama Pelatihan", "Nama_pelatihan", "Nama_Pelatihan", "Nama Program", "Program"],
        silabus_candidates=["Silabus", "Syllabus", "Materi Pelatihan", "Deskripsi Pelatihan"],
        unit_candidates=["Unit Kompetensi", "Unit_Kompetensi", "Unit_kompetensi", "UK"],
        level_candidates=["Level Kompetensi", "Level_Kompetensi", "Level_kompetensi", "Level"],
        source_label="Sheet ref",
        prioritas=1,
    )


def reference_from_labeled_histori(df_histori: Optional[pd.DataFrame]) -> pd.DataFrame:
    return _build_ref_dari_df(
        df_histori,
        nama_candidates=["Nama_pelatihan", "Nama Pelatihan", "Nama_Pelatihan"],
        silabus_candidates=["Silabus"],
        unit_candidates=["Unit_kompetensi", "Unit Kompetensi", "Unit_Kompetensi"],
        level_candidates=["Level_kompetensi", "Level Kompetensi", "Level_Kompetensi"],
        source_label="Histori sudah terisi",
        prioritas=2,
    )


def reference_from_standar(df_standar: Optional[pd.DataFrame]) -> pd.DataFrame:
    return _build_ref_dari_df(
        df_standar,
        nama_candidates=["Unit_Kompetensi", "Unit Kompetensi", "Unit_kompetensi"],
        silabus_candidates=["Deskripsi_Level", "Deskripsi Level", "Deskripsi"],
        unit_candidates=["Unit_Kompetensi", "Unit Kompetensi", "Unit_kompetensi"],
        level_candidates=["Level_Kompetensi", "Level Kompetensi", "Level_kompetensi"],
        source_label="Standar Kompetensi",
        prioritas=3,
        gabung_kolom_extra=[
            ["Jabatan"],
            ["Deskripsi_Level", "Deskripsi Level", "Deskripsi"],
        ],
    )


def build_reference_table(
    df_histori: Optional[pd.DataFrame],
    df_standar: Optional[pd.DataFrame],
    df_ref: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Gabungkan semua sumber referensi dengan prioritas."""
    ref_sheet   = reference_from_ref_sheet(df_ref)
    ref_histori = reference_from_labeled_histori(df_histori)
    ref_standar = reference_from_standar(df_standar)

    summary = {
        "jumlah_ref_sheet":   int(len(ref_sheet)),
        "jumlah_ref_histori": int(len(ref_histori)),
        "jumlah_ref_standar": int(len(ref_standar)),
    }

    parts = [df for df in [ref_sheet, ref_histori, ref_standar] if len(df) > 0]
    if not parts:
        return pd.DataFrame(columns=["nama", "silabus", "teks_ref", "unit", "level", "source", "prioritas", "_clean", "_clean_nama"]), summary

    df_ref_all = pd.concat(parts, ignore_index=True)

    # Operasi kolom, bukan apply per baris jika memungkinkan.
    # clean_text tetap perlu apply karena ada regex kompleks di dalamnya.
    df_ref_all["_clean"]      = df_ref_all["teks_ref"].apply(clean_text)
    df_ref_all["_clean_nama"] = df_ref_all["nama"].apply(clean_text)

    df_ref_all = df_ref_all[
        (df_ref_all["_clean"] != "")
        & (~df_ref_all["unit"].apply(is_empty_value))
    ].copy()

    df_ref_all = (
        df_ref_all
        .sort_values("prioritas")
        .drop_duplicates(subset=["_clean", "unit"], keep="first")
        .reset_index(drop=True)
    )

    return df_ref_all, summary


# ============================================================
# Keyword rule
# ============================================================

# Precompile rules sebagai konstanta modul — tidak dibuat ulang tiap panggilan.
_KEYWORD_RULES: List[Tuple[List[str], List[str]]] = [
    (
        ["data analytics", "data analitik", "analitik", "analytics",
         "statistika", "statistik", "olah data", "pengolahan data",
         "excel", "spreadsheet", "visualisasi", "dashboard",
         "power bi", "tableau", "python", "sql"],
        ["Pengolahan Data", "Penyajian Data Visual"],
    ),
    (
        ["informasi", "knowledge management", "manajemen data",
         "sistem informasi", "database", "taxpayer account", "coretax"],
        ["Manajemen Informasi", "Manjemen Informasi"],
    ),
    (
        ["aplikasi perkantoran", "office", "word", "powerpoint",
         "komputer", "digital literacy", "literasi digital"],
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
        ["regulasi", "peraturan", "undang", "hukum",
         "legal drafting", "disiplin pegawai"],
        ["Regulasi Peraturan Perundang-Undangan"],
    ),
    (
        ["pelayanan", "layanan publik", "komunikasi", "public speaking", "service"],
        ["Pelayanan Administratif"],
    ),
]


def _buat_lookup_standar(df_standar: Optional[pd.DataFrame]) -> Dict[str, Dict[str, str]]:
    """Buat lookup dict unit → {unit, level} dari standar kompetensi.

    Dibuat sekali, dipakai untuk semua baris. Lebih cepat dari iterrows
    tiap kali keyword_rule_predict dipanggil.
    """
    if df_standar is None:
        return {}

    unit_col  = find_col(df_standar, ["Unit_Kompetensi", "Unit Kompetensi", "Unit_kompetensi"], allow_contains=True)
    level_col = find_col(df_standar, ["Level_Kompetensi", "Level Kompetensi", "Level_kompetensi"], allow_contains=True)

    if unit_col is None:
        return {}

    lookup: Dict[str, Dict[str, str]] = {}
    for _, r in df_standar.iterrows():
        unit = str(r.get(unit_col, "")).strip()
        if not unit:
            continue
        level = str(r.get(level_col, "")).strip() if level_col else ""
        key   = clean_text(unit)
        if key not in lookup:
            lookup[key] = {"unit": unit, "level": level}

    return lookup


def keyword_rule_predict(
    text: str,
    standar_lookup: Dict[str, Dict[str, str]],
) -> Optional[Dict[str, str]]:
    """Prediksi via keyword rule menggunakan lookup dict.

    Signature berubah dari v2.2: menerima lookup dict, bukan df_standar,
    agar dict tidak dibuat ulang tiap baris.
    """
    text_clean = clean_text(text)
    if not text_clean:
        return None

    for keywords, target_units in _KEYWORD_RULES:
        if any(k in text_clean for k in keywords):
            for target in target_units:
                key = clean_text(target)
                if key in standar_lookup:
                    return standar_lookup[key]
            return {"unit": target_units[0], "level": ""}

    return None


# ============================================================
# [OPT-1] Vektorisasi assign hasil prediksi
# ============================================================

def _assign_hasil_vectorized(
    hasil: pd.DataFrame,
    similarities: np.ndarray,
    df_ref_all: pd.DataFrame,
    query_text: List[str],
    nama_col: str,
    silabus_col: str,
    unit_col: str,
    level_col: str,
    threshold_isi: float,
    threshold_review: float,
    metode_dipakai: str,
    standar_lookup: Dict[str, Dict[str, str]],
) -> Tuple[pd.DataFrame, int, int]:
    """Assign hasil prediksi ke DataFrame secara vektorikal.

    Strategi:
    1. Hitung best_idx dan best_score per baris dengan np.argmax / np.max
       — operasi NumPy penuh, tidak ada Python loop.
    2. Lakukan exact match via pd.merge (join) bukan filter per baris.
    3. Terapkan keyword rule hanya pada baris yang masih di bawah threshold.
    4. Assign semua kolom sekaligus dengan np.where / loc masking.

    Return: (hasil_df, jumlah_unit_diisi, jumlah_level_diisi)
    """
    n = len(hasil)
    idx = hasil.index  # pertahankan index asli

    # --------------------------------------------------------
    # Mask dasar
    # --------------------------------------------------------
    nama_series    = hasil[nama_col].fillna("").astype(str)
    silabus_series = hasil[silabus_col].fillna("").astype(str) if silabus_col in hasil.columns else pd.Series("", index=idx)
    unit_series    = hasil[unit_col].fillna("").astype(str)
    level_series   = hasil[level_col].fillna("").astype(str) if level_col in hasil.columns else pd.Series("", index=idx)

    mask_nama_kosong   = nama_series.apply(clean_text) == ""
    mask_sudah_isi     = (~unit_series.apply(is_empty_value)) & (~level_series.apply(is_empty_value))
    mask_perlu_prediksi = ~mask_nama_kosong & ~mask_sudah_isi

    # --------------------------------------------------------
    # [NumPy] Ambil best idx dan score dari matriks similarity
    # --------------------------------------------------------
    # similarities shape: (n_query, n_ref)
    best_idx_arr   = np.argmax(similarities, axis=1)   # shape (n,)
    best_score_arr = similarities[np.arange(n), best_idx_arr].astype(float)

    # Lookup nilai referensi berdasarkan best_idx (operasi kolom pd.Series)
    ref_unit_arr    = df_ref_all["unit"].iloc[best_idx_arr].values
    ref_level_arr   = df_ref_all["level"].iloc[best_idx_arr].values
    ref_source_arr  = df_ref_all["source"].iloc[best_idx_arr].values
    ref_nama_arr    = df_ref_all["nama"].iloc[best_idx_arr].values
    ref_silabus_arr = df_ref_all["silabus"].iloc[best_idx_arr].values

    # --------------------------------------------------------
    # [Pandas merge] Exact match nama — lebih cepat dari filter per baris
    # --------------------------------------------------------
    # Buat Series clean nama query
    clean_nama_query = nama_series.apply(clean_text).reset_index(drop=True)

    # Buat lookup: clean_nama → (unit, level, source, nama_asli, silabus_asli)
    ref_exact_nama = df_ref_all.drop_duplicates(subset=["_clean_nama"], keep="first").set_index("_clean_nama")

    # Lakukan lookup vektorikal
    match_unit_nama    = clean_nama_query.map(ref_exact_nama["unit"])
    match_level_nama   = clean_nama_query.map(ref_exact_nama["level"])
    match_source_nama  = clean_nama_query.map(ref_exact_nama["source"])
    match_nama_nama    = clean_nama_query.map(ref_exact_nama["nama"])
    match_silabus_nama = clean_nama_query.map(ref_exact_nama["silabus"])
    has_exact_nama     = match_unit_nama.notna().values  # bool array

    # Exact match nama + silabus (full)
    clean_full_query = pd.Series(query_text).reset_index(drop=True)
    ref_exact_full   = df_ref_all.drop_duplicates(subset=["_clean"], keep="first").set_index("_clean")

    match_unit_full    = clean_full_query.map(ref_exact_full["unit"])
    match_level_full   = clean_full_query.map(ref_exact_full["level"])
    match_source_full  = clean_full_query.map(ref_exact_full["source"])
    match_nama_full    = clean_full_query.map(ref_exact_full["nama"])
    match_silabus_full = clean_full_query.map(ref_exact_full["silabus"])
    has_exact_full     = match_unit_full.notna().values  # bool array

    # --------------------------------------------------------
    # Gabungkan: exact full > exact nama > similarity
    # --------------------------------------------------------
    final_unit    = np.where(has_exact_full, match_unit_full.fillna("").values,
                    np.where(has_exact_nama, match_unit_nama.fillna("").values,
                             ref_unit_arr))
    final_level   = np.where(has_exact_full, match_level_full.fillna("").values,
                    np.where(has_exact_nama, match_level_nama.fillna("").values,
                             ref_level_arr))
    final_source  = np.where(has_exact_full, (match_source_full.fillna("").values.astype(str) + " - Exact Match Full"),
                    np.where(has_exact_nama, (match_source_nama.fillna("").values.astype(str) + " - Exact Match Nama"),
                             ref_source_arr.astype(str)))
    final_nama    = np.where(has_exact_full, match_nama_full.fillna("").values,
                    np.where(has_exact_nama, match_nama_nama.fillna("").values,
                             ref_nama_arr))
    final_silabus = np.where(has_exact_full, match_silabus_full.fillna("").values,
                    np.where(has_exact_nama, match_silabus_nama.fillna("").values,
                             ref_silabus_arr))
    final_score   = np.where(has_exact_full | has_exact_nama, 1.0, best_score_arr)

    # --------------------------------------------------------
    # Keyword rule — hanya pada baris yang skor masih rendah
    # --------------------------------------------------------
    mask_perlu_kw = mask_perlu_prediksi.values & (final_score < threshold_review) & ~has_exact_full & ~has_exact_nama
    if mask_perlu_kw.any():
        teks_kw = (nama_series + " " + silabus_series).values
        for i in np.where(mask_perlu_kw)[0]:
            rule = keyword_rule_predict(teks_kw[i], standar_lookup)
            if rule is not None:
                final_unit[i]   = rule.get("unit", "")
                final_level[i]  = rule.get("level", "")
                final_source[i] = "Keyword Rule"
                final_nama[i]   = rule.get("unit", "")
                final_silabus[i]= ""
                final_score[i]  = max(final_score[i], 0.62)

    # --------------------------------------------------------
    # Hitung sumber_aman secara vektorikal
    # --------------------------------------------------------
    final_source_s = pd.Series(final_source)
    sumber_aman = (
        final_source_s.str.contains("Sheet ref", na=False)
        | final_source_s.str.contains("Histori", na=False)
        | final_source_s.str.contains("Exact Match", na=False)
        | final_source_s.str.contains("Keyword Rule", na=False)
    ).values

    # Standar kompetensi saja (tanpa exact match) hanya aman jika score >= 0.90
    standar_saja = final_source_s.str.contains("Standar Kompetensi", na=False).values & ~sumber_aman
    sumber_aman = sumber_aman | (standar_saja & (final_score >= 0.90))

    # --------------------------------------------------------
    # Tentukan status secara vektorikal
    # --------------------------------------------------------
    status_arr = np.full(n, STATUS_TIDAK_DIPREDIKSI, dtype=object)
    status_arr[mask_nama_kosong.values]  = STATUS_NAMA_KOSONG
    status_arr[mask_sudah_isi.values]    = STATUS_SUDAH_TERISI

    p = mask_perlu_prediksi.values
    can_isi  = p & sumber_aman & (final_score >= threshold_isi) & np.array([not is_empty_value(u) for u in final_unit])
    can_rev  = p & ~can_isi & (final_score >= threshold_review) & np.array([not is_empty_value(u) for u in final_unit])
    rendah   = p & ~can_isi & ~can_rev

    status_arr[can_isi] = STATUS_CONF_TINGGI
    status_arr[can_rev] = STATUS_REVIEW
    status_arr[rendah]  = STATUS_RENDAH

    # --------------------------------------------------------
    # Assign ke DataFrame — semua sekaligus, bukan per baris
    # --------------------------------------------------------
    hasil = hasil.copy()

    hasil["Metode_Prediksi"]   = metode_dipakai
    hasil["Nama_Pelatihan_Termirip"] = np.where(mask_perlu_prediksi.values, final_nama,    "")
    hasil["Silabus_Termirip"]        = np.where(mask_perlu_prediksi.values, final_silabus, "")
    hasil["Confidence_Prediksi"]     = np.where(
        mask_sudah_isi.values, 100.0,
        np.where(mask_perlu_prediksi.values, np.round(final_score * 100, 2), 0.0)
    )
    hasil["Sumber_Prediksi"] = np.where(
        mask_sudah_isi.values,  "Data asli",
        np.where(mask_perlu_prediksi.values, final_source, "")
    )
    hasil["Status_Prediksi"] = status_arr

    # Isi unit dan level hanya pada baris yang memang perlu diisi
    unit_sebelum  = unit_series.values.copy()
    level_sebelum = level_series.values.copy()

    unit_kosong_mask  = unit_series.apply(is_empty_value).values
    level_kosong_mask = level_series.apply(is_empty_value).values

    new_unit = np.where(
        can_isi & unit_kosong_mask, final_unit, unit_sebelum
    )
    new_level = np.where(
        can_isi & level_kosong_mask & np.array([not is_empty_value(l) for l in final_level]),
        final_level, level_sebelum
    )

    hasil["Prediksi_Unit_Kompetensi"]  = np.where(can_isi & unit_kosong_mask, final_unit, hasil["Prediksi_Unit_Kompetensi"])
    hasil["Prediksi_Level_Kompetensi"] = np.where(
        can_isi & level_kosong_mask & np.array([not is_empty_value(l) for l in final_level]),
        final_level, hasil["Prediksi_Level_Kompetensi"]
    )

    hasil[unit_col]  = new_unit
    hasil[level_col] = new_level

    jumlah_unit_diisi  = int((can_isi & unit_kosong_mask).sum())
    jumlah_level_diisi = int((can_isi & level_kosong_mask & np.array([not is_empty_value(l) for l in final_level])).sum())

    return hasil, jumlah_unit_diisi, jumlah_level_diisi


# ============================================================
# Fungsi utama prediksi (API publik, identik dengan v2.2)
# ============================================================

def prediksi_histori_unit_level(
    df_histori: pd.DataFrame,
    df_standar: pd.DataFrame,
    df_ref: Optional[pd.DataFrame] = None,
    batas_confidence: int = 75,
    metode_prediksi: str = "SBERT",
    batch_size: int = BATCH_SIZE_DEFAULT,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Prediksi Unit_kompetensi dan Level_kompetensi pada histori.

    API identik dengan v2.2 kecuali ada parameter tambahan `batch_size`.
    Parameter tambahan ini opsional — default 512 cocok untuk hampir semua kasus.

    Perubahan internal dari v2.2:
    - [OPT-1] Loop per baris diganti operasi kolom NumPy/Pandas.
    - [OPT-2] Embedding referensi di-cache di session_state.
    - [OPT-3] Query di-encode per batch.
    """
    hasil = df_histori.copy().reset_index(drop=True)

    # Temukan kolom.
    nama_col    = find_col(hasil, ["Nama_pelatihan", "Nama Pelatihan", "Nama_Pelatihan"], allow_contains=True)
    silabus_col = find_col(hasil, ["Silabus"], allow_contains=True)
    unit_col    = find_col(hasil, ["Unit_kompetensi", "Unit Kompetensi", "Unit_Kompetensi"], allow_contains=True)
    level_col   = find_col(hasil, ["Level_kompetensi", "Level Kompetensi", "Level_Kompetensi"], allow_contains=True)

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

    if unit_col is None:
        hasil["Unit_kompetensi"] = ""
        unit_col = "Unit_kompetensi"

    if level_col is None:
        hasil["Level_kompetensi"] = ""
        level_col = "Level_kompetensi"

    if silabus_col is None:
        hasil["Silabus"] = ""
        silabus_col = "Silabus"

    unit_kosong_awal  = int(hasil[unit_col].apply(is_empty_value).sum())
    level_kosong_awal = int(hasil[level_col].apply(is_empty_value).sum())

    # Kolom audit — inisialisasi sekaligus.
    hasil["Unit_Asli"]               = hasil[unit_col]
    hasil["Level_Asli"]              = hasil[level_col]
    hasil["Prediksi_Unit_Kompetensi"]  = hasil[unit_col]
    hasil["Prediksi_Level_Kompetensi"] = hasil[level_col]
    hasil["Nama_Pelatihan_Termirip"]   = ""
    hasil["Silabus_Termirip"]          = ""
    hasil["Confidence_Prediksi"]       = 0.0
    hasil["Sumber_Prediksi"]           = ""
    hasil["Metode_Prediksi"]           = ""
    hasil["Status_Prediksi"]           = STATUS_TIDAK_DIPREDIKSI

    # Bangun tabel referensi.
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

    threshold_isi    = float(batas_confidence) / 100.0
    threshold_review = THRESHOLD_REVIEW_DEFAULT

    # Bangun teks query.
    query_text = (
        hasil[nama_col].fillna("").astype(str)
        + " "
        + hasil[silabus_col].fillna("").astype(str)
    ).apply(clean_text).tolist()

    ref_text = df_ref_all["_clean"].tolist()

    # [OPT-2] + [OPT-3] Hitung similarity matrix.
    similarities, metode_dipakai = build_similarity_matrix(
        query_text=query_text,
        ref_text=ref_text,
        metode_prediksi=metode_prediksi,
        batch_size=batch_size,
    )

    # [OPT-1] Buat lookup standar sekali saja.
    standar_lookup = _buat_lookup_standar(df_standar)

    # [OPT-1] Assign vektorikal.
    hasil, jumlah_unit_diisi, jumlah_level_diisi = _assign_hasil_vectorized(
        hasil=hasil,
        similarities=similarities,
        df_ref_all=df_ref_all,
        query_text=query_text,
        nama_col=nama_col,
        silabus_col=silabus_col,
        unit_col=unit_col,
        level_col=level_col,
        threshold_isi=threshold_isi,
        threshold_review=threshold_review,
        metode_dipakai=metode_dipakai,
        standar_lookup=standar_lookup,
    )

    unit_kosong_akhir  = int(hasil[unit_col].apply(is_empty_value).sum())
    level_kosong_akhir = int(hasil[level_col].apply(is_empty_value).sum())

    jumlah_tinggi = int((hasil["Status_Prediksi"] == STATUS_CONF_TINGGI).sum())
    jumlah_review = int((hasil["Status_Prediksi"] == STATUS_REVIEW).sum())
    jumlah_rendah = int((hasil["Status_Prediksi"] == STATUS_RENDAH).sum())
    jumlah_sudah  = int((hasil["Status_Prediksi"] == STATUS_SUDAH_TERISI).sum())

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


# Alias kompatibilitas
prediksi_histori_unit_level_sbert  = prediksi_histori_unit_level
prediksi_histori_unit_level_tfidf  = prediksi_histori_unit_level
