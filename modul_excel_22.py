# modul_excel_22.py
# ============================================================
# Modul Excel untuk Dashboard Evaluasi Kompetensi Pegawai v2.2
# ============================================================
#
# Fungsi utama modul ini:
# 1. Membaca file Excel dengan aman.
# 2. Menormalisasi nama kolom agar konsisten.
# 3. Mendeteksi baris header jika file Excel punya baris judul di atas tabel.
# 4. Memastikan NIP dibaca sebagai teks.
# 5. Menyediakan helper validasi sheet dan kolom.
# 6. Membuat diagnostik kualitas data dasar.
#
# Catatan untuk pemula:
# Modul ini tidak menghitung evaluasi kompetensi.
# Modul ini hanya fokus pada input Excel.

from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd


# ============================================================
# Konstanta standar Excel v2.2
# ============================================================

SHEET_DATA_PEGAWAI = "Data Pegawai"
SHEET_STANDAR_KOMPETENSI = "Standar Kompetensi"
SHEET_HISTORI_PELATIHAN = "Histori Pelatihan"
SHEET_REF = "ref"
SHEET_HASIL_EVALUASI = "Hasil Evaluasi"

SHEET_WAJIB = [
    SHEET_DATA_PEGAWAI,
    SHEET_STANDAR_KOMPETENSI,
    SHEET_HISTORI_PELATIHAN,
]

SHEET_OPSIONAL = [
    SHEET_REF,
    SHEET_HASIL_EVALUASI,
]


# Kolom standar sesuai template terbaru.
KOLOM_STANDAR = {
    SHEET_DATA_PEGAWAI: [
        "No",
        "Nama_Pegawai",
        "NIP_Panjang",
        "Jabatan",
        "Unit_Es_IV",
        "Unit_Es_III",
        "Unit_Es_II",
    ],
    SHEET_STANDAR_KOMPETENSI: [
        "No",
        "Jabatan",
        "Unit_Kompetensi",
        "Level_Kompetensi",
        "Deskripsi_Level",
    ],
    SHEET_HISTORI_PELATIHAN: [
        "No",
        "Nama_Pegawai",
        "NIP_Panjang",
        "Nama_pelatihan",
        "Silabus",
        "Unit_kompetensi",
        "Level_kompetensi",
        "Metode_Pelatihan",
    ],
    SHEET_REF: [
        "Nama_Pelatihan",
        "Silabus",
        "Unit_Kompetensi",
        "Level_Kompetensi",
    ],
}


# Alias membantu aplikasi tetap membaca variasi kecil nama kolom.
# Misalnya "Nama Pegawai" tetap dikenali sebagai "Nama_Pegawai".
ALIASES_KOLOM = {
    "nama_pegawai": [
        "Nama_Pegawai",
        "Nama Pegawai",
        "Nama",
        "Pegawai",
    ],
    "nip_panjang": [
        "NIP_Panjang",
        "NIP Panjang",
        "NIP",
        "Nip",
        "Nomor Induk Pegawai",
    ],
    "jabatan": [
        "Jabatan",
        "Nama Jabatan",
    ],
    "unit_es_iv": [
        "Unit_Es_IV",
        "Unit Es IV",
        "Unit Eselon IV",
        "Seksi",
        "Seksi / Unit",
    ],
    "unit_es_iii": [
        "Unit_Es_III",
        "Unit Es III",
        "Unit Eselon III",
    ],
    "unit_es_ii": [
        "Unit_Es_II",
        "Unit Es II",
        "Unit Eselon II",
        "Kantor",
    ],
    "nama_pelatihan": [
        "Nama_pelatihan",
        "Nama Pelatihan",
        "Nama_Pelatihan",
        "Nama Program",
        "Program",
        "Pelatihan",
    ],
    "silabus": [
        "Silabus",
        "Syllabus",
        "Materi",
        "Materi Pelatihan",
        "Deskripsi Pelatihan",
    ],
    "unit_kompetensi": [
        "Unit_kompetensi",
        "Unit Kompetensi",
        "Unit_Kompetensi",
        "UK",
        "Kompetensi",
    ],
    "level_kompetensi": [
        "Level_kompetensi",
        "Level Kompetensi",
        "Level_Kompetensi",
        "Level",
    ],
    "deskripsi_level": [
        "Deskripsi_Level",
        "Deskripsi Level",
        "Deskripsi",
        "Uraian Level",
    ],
    "metode_pelatihan": [
        "Metode_Pelatihan",
        "Metode Pelatihan",
        "Metode",
    ],
}


# ============================================================
# Normalisasi teks dan nama kolom
# ============================================================

def normalize_col_name(col) -> str:
    """Mengubah nama kolom menjadi bentuk aman.

    Contoh:
    - "Unit Kompetensi" -> "Unit_Kompetensi"
    - "No." -> "No"
    - " Level Kompetensi " -> "Level_Kompetensi"
    """
    col = "" if col is None else str(col).strip()
    col = re.sub(r"[\n\r\t]+", " ", col)
    col = col.replace(".", "")
    col = re.sub(r"\s+", "_", col)
    return col


def make_unique_columns(columns: Iterable) -> list[str]:
    """Membuat nama kolom unik agar tidak bentrok.

    Jika ada dua kolom bernama sama, hasilnya menjadi:
    - Kolom
    - Kolom_2
    """
    hasil = []
    counter: Dict[str, int] = {}

    for col in columns:
        base = normalize_col_name(col)

        if base == "" or base.lower().startswith("unnamed") or base.lower() == "nan":
            base = "Kolom_Tanpa_Nama"

        counter[base] = counter.get(base, 0) + 1

        if counter[base] == 1:
            hasil.append(base)
        else:
            hasil.append(f"{base}_{counter[base]}")

    return hasil


def normalisasi_kolom(df: pd.DataFrame) -> pd.DataFrame:
    """Menormalisasi seluruh nama kolom dataframe."""
    df = df.copy()
    df.columns = make_unique_columns(df.columns)
    return df


def clean_text(text) -> str:
    """Membersihkan teks untuk kebutuhan pencarian dan ML."""
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"[^a-z0-9À-ÿ\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_empty_value(x) -> bool:
    """Mengecek apakah nilai dianggap kosong."""
    if pd.isna(x):
        return True

    text = str(x).strip()

    return text == "" or text == "-" or text.lower() in ["nan", "none", "null", "nat"]


def normalize_id_value(x) -> str:
    """Menormalkan NIP/ID agar selalu aman sebagai teks.

    Excel sering membuat NIP menjadi angka float, misalnya:
    198001012010011001.0

    Fungsi ini menghapus akhiran .0 jika memang angka bulat.
    """
    if pd.isna(x):
        return ""

    if isinstance(x, (int, np.integer)):
        return str(int(x))

    if isinstance(x, (float, np.floating)):
        if np.isfinite(x) and float(x).is_integer():
            return str(int(x))
        return str(x).strip()

    text = str(x).strip()

    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]

    return text


def force_text_key(df: Optional[pd.DataFrame], col):
    """Memaksa kolom tertentu menjadi teks.

    Biasanya dipakai untuk NIP_Panjang.
    """
    if df is not None and col is not None and col in df.columns:
        df[col] = df[col].apply(normalize_id_value).astype(str)

    return df


# ============================================================
# Pencarian kolom fleksibel
# ============================================================

def _norm_key(text: str) -> str:
    return normalize_col_name(text).lower()


def expand_candidates(candidates: list[str]) -> list[str]:
    """Menambah kandidat kolom berdasarkan alias yang dikenal."""
    expanded = []

    for cand in candidates:
        expanded.append(cand)
        key = _norm_key(cand)

        for alias_key, alias_values in ALIASES_KOLOM.items():
            alias_norms = [_norm_key(x) for x in alias_values]
            if key == alias_key or key in alias_norms:
                expanded.extend(alias_values)

    # Hilangkan duplikasi sambil menjaga urutan.
    hasil = []
    seen = set()
    for item in expanded:
        norm = _norm_key(item)
        if norm not in seen:
            hasil.append(item)
            seen.add(norm)

    return hasil


def find_col(
    df: Optional[pd.DataFrame],
    candidates: list[str],
    allow_contains: bool = False,
):
    """Mencari kolom dataframe berdasarkan beberapa kandidat nama.

    Parameter:
    - df: dataframe yang ingin dicek.
    - candidates: daftar nama kolom kemungkinan.
    - allow_contains: jika True, boleh cocok sebagian nama.
    """
    if df is None:
        return None

    cols = list(df.columns)
    norm_map = {_norm_key(c): c for c in cols}
    expanded_candidates = expand_candidates(candidates)

    # Cocok persis setelah normalisasi.
    for cand in expanded_candidates:
        key = _norm_key(cand)
        if key in norm_map:
            return norm_map[key]

    # Cocok sebagian jika diizinkan.
    if allow_contains:
        for cand in expanded_candidates:
            key = _norm_key(cand)
            for col in cols:
                ckey = _norm_key(col)
                if key in ckey or ckey in key:
                    return col

    return None


def cek_kolom_wajib(
    df: pd.DataFrame,
    mapping_wajib: Dict[str, list[str]],
) -> Dict[str, Any]:
    """Memeriksa apakah kolom wajib tersedia.

    mapping_wajib contoh:
    {
        "Nama_Pegawai": ["Nama_Pegawai", "Nama Pegawai"],
        "NIP_Panjang": ["NIP_Panjang", "NIP"],
    }
    """
    hasil = {
        "lengkap": True,
        "kolom_ditemukan": {},
        "kolom_kurang": [],
    }

    for nama_standar, candidates in mapping_wajib.items():
        col = find_col(df, candidates, allow_contains=True)
        hasil["kolom_ditemukan"][nama_standar] = col

        if col is None:
            hasil["lengkap"] = False
            hasil["kolom_kurang"].append(nama_standar)

    return hasil


# ============================================================
# Pembacaan sheet Excel
# ============================================================

def get_sheet_names(uploaded_file) -> list[str]:
    """Mengambil daftar sheet dari file Excel."""
    uploaded_file.seek(0)
    xls = pd.ExcelFile(uploaded_file)
    return xls.sheet_names


def sheet_exists(uploaded_file, sheet_name: str) -> bool:
    """Mengecek apakah sheet ada dalam workbook."""
    return sheet_name in get_sheet_names(uploaded_file)


def validasi_sheet_wajib(uploaded_file) -> Dict[str, Any]:
    """Memvalidasi keberadaan sheet wajib."""
    sheet_names = get_sheet_names(uploaded_file)
    missing = [s for s in SHEET_WAJIB if s not in sheet_names]

    return {
        "valid": len(missing) == 0,
        "sheet_names": sheet_names,
        "sheet_wajib": SHEET_WAJIB,
        "sheet_kurang": missing,
    }


def detect_header_row(
    uploaded_file,
    sheet_name: str,
    preferred_cols: list[str],
    max_scan: int = 25,
) -> Tuple[int, int]:
    """Mendeteksi baris header terbaik.

    Berguna jika file Excel punya judul atau catatan di atas tabel.
    Fungsi ini membaca maksimal 25 baris awal, lalu mencari baris
    yang paling banyak mengandung nama kolom pilihan.
    """
    uploaded_file.seek(0)
    preview = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None, nrows=max_scan)

    preferred_expanded = expand_candidates(preferred_cols)
    preferred = {_norm_key(c) for c in preferred_expanded}

    best_row = 0
    best_score = -1

    for idx, row in preview.iterrows():
        values = {_norm_key(v) for v in row.dropna().tolist()}
        score = len(values.intersection(preferred))

        if score > best_score:
            best_score = score
            best_row = idx

    return int(best_row), int(best_score)


def baca_sheet_otomatis(
    uploaded_file,
    sheet_name: str,
    preferred_cols: list[str],
):
    """Membaca sheet dengan deteksi header otomatis.

    Return:
    - df
    - nomor header Excel 1-based
    - score kecocokan header
    """
    header_row, score = detect_header_row(uploaded_file, sheet_name, preferred_cols)

    uploaded_file.seek(0)
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header_row)
    df = normalisasi_kolom(df)
    df = df.dropna(how="all").reset_index(drop=True)

    return df, header_row + 1, score


def baca_sheet(
    uploaded_file,
    sheet_name: str,
    header: int = 0,
) -> pd.DataFrame:
    """Membaca sheet dengan header tertentu."""
    uploaded_file.seek(0)
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header)
    return normalisasi_kolom(df)


def baca_sheet_standar(
    uploaded_file,
    sheet_name: str,
) -> tuple[pd.DataFrame, int, int]:
    """Membaca sheet berdasarkan daftar kolom standar v2.2."""
    preferred_cols = KOLOM_STANDAR.get(sheet_name, [])
    return baca_sheet_otomatis(uploaded_file, sheet_name, preferred_cols)


def pilih_sheet_dengan_kolom(
    uploaded_file,
    kandidat_sheet: list[str],
    preferred_cols: list[str],
) -> str:
    """Memilih sheet terbaik berdasarkan kandidat nama dan kecocokan kolom."""
    sheet_names = get_sheet_names(uploaded_file)

    for sheet in kandidat_sheet:
        if sheet in sheet_names:
            return sheet

    best_sheet = sheet_names[0]
    best_score = -1

    for sheet in sheet_names:
        _, score = detect_header_row(uploaded_file, sheet, preferred_cols)
        if score > best_score:
            best_score = score
            best_sheet = sheet

    return best_sheet


# ============================================================
# Diagnostik kualitas data
# ============================================================

def _count_empty(df: Optional[pd.DataFrame], col) -> int:
    if df is None or col is None or col not in df.columns:
        return 0
    return int(df[col].apply(is_empty_value).sum())


def _count_duplicate_non_empty(df: Optional[pd.DataFrame], col) -> int:
    if df is None or col is None or col not in df.columns:
        return 0

    s = df[col].apply(normalize_id_value).astype(str)
    s = s[~s.apply(is_empty_value)]

    if len(s) == 0:
        return 0

    return int(s.duplicated(keep=False).sum())


def diagnostik_dataframe(
    df: Optional[pd.DataFrame],
    jenis: str,
) -> Dict[str, Any]:
    """Membuat ringkasan kualitas satu dataframe.

    jenis:
    - "pegawai"
    - "standar"
    - "histori"
    - "ref"
    """
    if df is None:
        return {
            "tersedia": False,
            "jumlah_baris": 0,
            "catatan": "Data tidak tersedia.",
        }

    hasil: Dict[str, Any] = {
        "tersedia": True,
        "jumlah_baris": int(len(df)),
        "jumlah_kolom": int(len(df.columns)),
        "kolom": list(df.columns),
    }

    if jenis == "pegawai":
        col_nama = find_col(df, ["Nama_Pegawai", "Nama Pegawai"], allow_contains=True)
        col_nip = find_col(df, ["NIP_Panjang", "NIP"], allow_contains=True)
        col_jabatan = find_col(df, ["Jabatan"], allow_contains=True)

        hasil.update({
            "nama_kosong": _count_empty(df, col_nama),
            "nip_kosong": _count_empty(df, col_nip),
            "nip_duplikat": _count_duplicate_non_empty(df, col_nip),
            "jabatan_kosong": _count_empty(df, col_jabatan),
        })

    elif jenis == "standar":
        col_jabatan = find_col(df, ["Jabatan"], allow_contains=True)
        col_unit = find_col(df, ["Unit_Kompetensi", "Unit Kompetensi"], allow_contains=True)
        col_level = find_col(df, ["Level_Kompetensi", "Level Kompetensi"], allow_contains=True)

        hasil.update({
            "jabatan_kosong": _count_empty(df, col_jabatan),
            "unit_kosong": _count_empty(df, col_unit),
            "level_kosong": _count_empty(df, col_level),
        })

    elif jenis == "histori":
        col_nama = find_col(df, ["Nama_Pegawai", "Nama Pegawai"], allow_contains=True)
        col_nip = find_col(df, ["NIP_Panjang", "NIP"], allow_contains=True)
        col_pelatihan = find_col(df, ["Nama_pelatihan", "Nama Pelatihan"], allow_contains=True)
        col_silabus = find_col(df, ["Silabus"], allow_contains=True)
        col_unit = find_col(df, ["Unit_kompetensi", "Unit Kompetensi"], allow_contains=True)
        col_level = find_col(df, ["Level_kompetensi", "Level Kompetensi"], allow_contains=True)

        hasil.update({
            "nama_pegawai_kosong": _count_empty(df, col_nama),
            "nip_kosong": _count_empty(df, col_nip),
            "nama_pelatihan_kosong": _count_empty(df, col_pelatihan),
            "silabus_kosong": _count_empty(df, col_silabus),
            "unit_kosong": _count_empty(df, col_unit),
            "level_kosong": _count_empty(df, col_level),
        })

    elif jenis == "ref":
        col_pelatihan = find_col(df, ["Nama Pelatihan", "Nama_pelatihan"], allow_contains=True)
        col_silabus = find_col(df, ["Silabus"], allow_contains=True)
        col_unit = find_col(df, ["Unit Kompetensi", "Unit_Kompetensi"], allow_contains=True)
        col_level = find_col(df, ["Level Kompetensi", "Level_Kompetensi"], allow_contains=True)

        hasil.update({
            "nama_pelatihan_kosong": _count_empty(df, col_pelatihan),
            "silabus_kosong": _count_empty(df, col_silabus),
            "unit_kosong": _count_empty(df, col_unit),
            "level_kosong": _count_empty(df, col_level),
        })

        if col_pelatihan is not None and col_pelatihan in df.columns:
            s = df[col_pelatihan].astype(str).str.strip()
            s = s[~s.apply(is_empty_value)]
            hasil["nama_pelatihan_duplikat"] = int(s.duplicated(keep=False).sum())
        else:
            hasil["nama_pelatihan_duplikat"] = 0

    return hasil


def buat_diagnostik_excel(
    df_pegawai: Optional[pd.DataFrame] = None,
    df_standar: Optional[pd.DataFrame] = None,
    df_histori: Optional[pd.DataFrame] = None,
    df_ref: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Membuat diagnostik kualitas data untuk seluruh workbook."""
    return {
        "Data Pegawai": diagnostik_dataframe(df_pegawai, "pegawai"),
        "Standar Kompetensi": diagnostik_dataframe(df_standar, "standar"),
        "Histori Pelatihan": diagnostik_dataframe(df_histori, "histori"),
        "ref": diagnostik_dataframe(df_ref, "ref"),
    }


def ringkas_workbook(uploaded_file) -> Dict[str, Any]:
    """Membuat ringkasan workbook tanpa memproses evaluasi."""
    sheet_names = get_sheet_names(uploaded_file)
    validasi = validasi_sheet_wajib(uploaded_file)

    return {
        "sheet_names": sheet_names,
        "validasi_sheet": validasi,
        "jumlah_sheet": len(sheet_names),
    }


# ============================================================
# Helper tampilan pesan error
# ============================================================

def format_kolom_kurang(sheet_name: str, kolom_kurang: list[str]) -> str:
    """Membuat pesan kolom kurang yang mudah dibaca."""
    if not kolom_kurang:
        return ""

    bullet = "\n".join([f"- {k}" for k in kolom_kurang])
    return f"Kolom wajib belum ditemukan pada sheet '{sheet_name}':\n{bullet}"


def buat_pesan_sheet_kurang(sheet_kurang: list[str]) -> str:
    """Membuat pesan sheet kurang yang mudah dibaca."""
    if not sheet_kurang:
        return ""

    bullet = "\n".join([f"- {s}" for s in sheet_kurang])
    return f"Sheet wajib belum ditemukan:\n{bullet}"
