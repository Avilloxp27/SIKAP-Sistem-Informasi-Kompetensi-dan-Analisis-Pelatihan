# modul_evaluasi_22.py
# ============================================================
# Modul Evaluasi Kompetensi Pegawai v2.2
# ============================================================
#
# Fungsi utama modul ini:
# 1. Membaca sheet Excel standar.
# 2. Memanggil prediksi ML jika diaktifkan.
# 3. Menjaga data asli agar tidak ditimpa sembarangan.
# 4. Menghitung status kompetensi pegawai.
# 5. Membuat insight gap dan rekomendasi pelatihan.
# 6. Membuat metadata dan diagnostik kualitas data.
# 7. Mengubah dataframe menjadi file Excel untuk download.
#
# Catatan untuk pemula:
# Modul ini adalah "mesin proses" aplikasi.
# File app_2.2.py hanya mengatur tampilan Streamlit.

from __future__ import annotations

from io import BytesIO
from typing import Optional, Dict, Any, List

import pandas as pd
import sys

# Alias ini membantu jika ada modul lama yang masih mengimpor "modul_evaluasi".
sys.modules.setdefault("modul_evaluasi", sys.modules[__name__])

from modul_excel_22 import (
    SHEET_DATA_PEGAWAI,
    SHEET_STANDAR_KOMPETENSI,
    SHEET_HISTORI_PELATIHAN,
    SHEET_REF,
    SHEET_HASIL_EVALUASI,
    normalisasi_kolom,
    find_col,
    force_text_key,
    baca_sheet_otomatis,
    validasi_sheet_wajib,
    is_empty_value,
    buat_diagnostik_excel,
    format_kolom_kurang,
    buat_pesan_sheet_kurang,
)

from modul_prediksi_23 import prediksi_histori_unit_level


# ============================================================
# Helper export
# ============================================================

def convert_df_to_excel(dataframe: pd.DataFrame) -> bytes:
    """Mengubah dataframe menjadi file Excel dalam bentuk bytes.

    Fungsi ini dipakai oleh tombol download Streamlit.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Hasil Evaluasi")
    return output.getvalue()


# ============================================================
# Helper teks gabungan
# ============================================================

def gabung_unik(series) -> str:
    """Menggabungkan nilai unik dalam series dengan pemisah ' | '."""
    data = []

    for x in series.dropna():
        x = str(x).strip()
        if x and x.lower() != "nan" and x != "-":
            data.append(x)

    data = sorted(set(data))

    return " | ".join(data) if data else "-"


def pecah_item(teks) -> list[str]:
    """Memecah teks gabungan 'a | b | c' menjadi list item."""
    if pd.isna(teks):
        return []

    teks = str(teks).strip()
    if teks == "" or teks == "-" or teks.lower() == "nan":
        return []

    hasil = []
    for item in teks.split(" | "):
        item = item.strip()
        if item and item != "-":
            hasil.append(item)

    return hasil


def bersihkan_level_item(item) -> str:
    """Membersihkan tampilan item kompetensi.

    Contoh:
    'Analisis Kebijakan (butuh level 2, saat ini 0)'
    menjadi:
    'Analisis Kebijakan lv 2'
    """
    item = str(item).strip()

    if "(butuh level" in item:
        nama = item.split("(butuh level")[0].strip()
        try:
            level = item.split("butuh level")[1].split(",")[0].strip()
            return f"{nama} lv {level}"
        except Exception:
            return nama

    return item


def bersihkan_nama_unit_gap(item) -> str:
    """Mengambil nama unit kompetensi saja dari teks gap."""
    item = str(item).strip()

    if "(butuh level" in item:
        item = item.split("(butuh level")[0].strip()

    return item


def hitung_top_item(
    df: Optional[pd.DataFrame],
    kolom: str,
    top_n: int = 5,
    bersihkan_gap: bool = False,
) -> pd.DataFrame:
    """Menghitung item terbanyak dari kolom gabungan.

    Cocok untuk:
    - Top gap kompetensi
    - Top rekomendasi pelatihan
    """
    if df is None or kolom not in df.columns:
        return pd.DataFrame(columns=["Item", "Jumlah"])

    items = []

    for teks in df[kolom].dropna():
        for item in pecah_item(teks):
            item = item.strip()
            if not item or item == "-" or item.lower() == "belum ada rekomendasi pelatihan":
                continue

            if bersihkan_gap:
                item = bersihkan_nama_unit_gap(item)

            items.append(item)

    if not items:
        return pd.DataFrame(columns=["Item", "Jumlah"])

    top = pd.Series(items).value_counts().head(top_n).reset_index()
    top.columns = ["Item", "Jumlah"]

    return top


# ============================================================
# Helper validasi dan diagnostik
# ============================================================

def _cek_col_or_raise(
    df: pd.DataFrame,
    sheet_name: str,
    mapping: Dict[str, List[str]],
) -> Dict[str, str]:
    """Mencari kolom wajib dan raise error jika ada yang kurang."""
    found: Dict[str, str] = {}
    missing: List[str] = []

    for label, candidates in mapping.items():
        col = find_col(df, candidates, allow_contains=True)
        if col is None:
            missing.append(label)
        else:
            found[label] = col

    if missing:
        raise ValueError(format_kolom_kurang(sheet_name, missing))

    return found


def _diagnostik_ringkas(meta_diag: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Mengubah output diagnostik Excel menjadi format ringkas untuk app_2.2.py.

    app_2.2.py menampilkan meta['diagnostik_data'] dengan key:
    - pegawai
    - standar
    - histori
    - ref
    """
    peg = meta_diag.get("Data Pegawai", {}) or {}
    std = meta_diag.get("Standar Kompetensi", {}) or {}
    his = meta_diag.get("Histori Pelatihan", {}) or {}
    ref = meta_diag.get("ref", {}) or {}

    return {
        "pegawai": {
            "Baris": peg.get("jumlah_baris", 0),
            "NIP kosong": peg.get("nip_kosong", 0),
            "NIP duplikat": peg.get("nip_duplikat", 0),
            "Nama kosong": peg.get("nama_kosong", 0),
            "Jabatan kosong": peg.get("jabatan_kosong", 0),
        },
        "standar": {
            "Baris": std.get("jumlah_baris", 0),
            "Jabatan kosong": std.get("jabatan_kosong", 0),
            "Unit kosong": std.get("unit_kosong", 0),
            "Level kosong": std.get("level_kosong", 0),
        },
        "histori": {
            "Baris": his.get("jumlah_baris", 0),
            "Nama pegawai kosong": his.get("nama_pegawai_kosong", 0),
            "NIP kosong": his.get("nip_kosong", 0),
            "Nama pelatihan kosong": his.get("nama_pelatihan_kosong", 0),
            "Silabus kosong": his.get("silabus_kosong", 0),
            "Unit kosong": his.get("unit_kosong", 0),
            "Level kosong": his.get("level_kosong", 0),
        },
        "ref": {
            "Baris": ref.get("jumlah_baris", 0),
            "Nama pelatihan kosong": ref.get("nama_pelatihan_kosong", 0),
            "Silabus kosong": ref.get("silabus_kosong", 0),
            "Unit kosong": ref.get("unit_kosong", 0),
            "Level kosong": ref.get("level_kosong", 0),
            "Nama duplikat": ref.get("nama_pelatihan_duplikat", 0),
        },
    }


def _ensure_optional_column(df: pd.DataFrame, col_name: str, default_value="") -> pd.DataFrame:
    """Menambahkan kolom opsional jika belum tersedia."""
    if col_name not in df.columns:
        df[col_name] = default_value
    return df


def _strip_text_column(df: pd.DataFrame, col: Optional[str]) -> pd.DataFrame:
    """Membersihkan spasi awal/akhir pada kolom teks."""
    if col is not None and col in df.columns:
        df[col] = df[col].fillna("").astype(str).str.strip()
    return df


# ============================================================
# Fungsi utama proses upload
# ============================================================

def proses_file_upload(
    uploaded_file,
    aktifkan_prediksi: bool = True,
    batas_confidence: int = 75,
    isi_level_default: bool = False,
    nilai_level_default: int = 1,
    metode_prediksi: str = "SBERT",
):
    """Membaca dan memproses file Excel menjadi hasil evaluasi.

    Parameter:
    - uploaded_file: file Excel dari Streamlit atau BytesIO.
    - aktifkan_prediksi: True jika ML ingin dijalankan.
    - batas_confidence: batas confidence untuk isi otomatis.
    - isi_level_default: jika True, level kosong diisi nilai default.
    - nilai_level_default: nilai level default.
    - metode_prediksi: 'SBERT' atau 'TF-IDF'.

    Return:
    - hasil: dataframe hasil evaluasi pegawai.
    - meta: metadata proses untuk ditampilkan di aplikasi.
    """
    uploaded_file.seek(0)

    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names

    meta: Dict[str, Any] = {
        "sheet_names": sheet_names,
        "catatan": [],
    }

    # ------------------------------------------------------------
    # Mode khusus: jika file sudah punya Hasil Evaluasi, langsung baca.
    # ------------------------------------------------------------
    if SHEET_HASIL_EVALUASI in sheet_names:
        df_hasil, header_hasil, _ = baca_sheet_otomatis(
            uploaded_file,
            SHEET_HASIL_EVALUASI,
            [
                "Nama_Pegawai",
                "NIP_Panjang",
                "Jabatan",
                "Status_Kompetensi",
                "Skor_Kecocokan_%",
            ],
        )

        meta["mode"] = "Membaca sheet Hasil Evaluasi"
        meta["metode_prediksi"] = "Tidak berjalan - file sudah berisi Hasil Evaluasi"
        meta["header_excel"] = {SHEET_HASIL_EVALUASI: header_hasil}
        meta["catatan"].append("Sheet Hasil Evaluasi ditemukan, aplikasi langsung membaca hasil tanpa menghitung ulang.")
        meta["diagnostik_data"] = {
            "pegawai": {"Baris": len(df_hasil)},
            "standar": {},
            "histori": {},
            "ref": {},
        }

        return df_hasil, meta

    # ------------------------------------------------------------
    # Validasi sheet wajib.
    # ------------------------------------------------------------
    validasi_sheet = validasi_sheet_wajib(uploaded_file)
    if not validasi_sheet["valid"]:
        raise ValueError(buat_pesan_sheet_kurang(validasi_sheet["sheet_kurang"]))

    # ------------------------------------------------------------
    # Baca sheet dengan deteksi header otomatis.
    # ------------------------------------------------------------
    df_pegawai, header_pegawai, _ = baca_sheet_otomatis(
        uploaded_file,
        SHEET_DATA_PEGAWAI,
        [
            "No.",
            "Nama_Pegawai",
            "NIP_Panjang",
            "Jabatan",
            "Unit_Es_IV",
            "Unit_Es_III",
            "Unit_Es_II",
        ],
    )

    df_standar, header_standar, _ = baca_sheet_otomatis(
        uploaded_file,
        SHEET_STANDAR_KOMPETENSI,
        [
            "No.",
            "Jabatan",
            "Unit Kompetensi",
            "Unit_Kompetensi",
            "Level Kompetensi",
            "Level_Kompetensi",
            "Deskripsi Level",
        ],
    )

    df_histori, header_histori, _ = baca_sheet_otomatis(
        uploaded_file,
        SHEET_HISTORI_PELATIHAN,
        [
            "No.",
            "Nama_Pegawai",
            "NIP_Panjang",
            "Nama_pelatihan",
            "Nama Pelatihan",
            "Silabus",
            "Unit_kompetensi",
            "Unit Kompetensi",
            "Level_kompetensi",
            "Level Kompetensi",
            "Metode_Pelatihan",
        ],
    )

    df_ref = None
    header_ref = None
    if SHEET_REF in sheet_names:
        df_ref, header_ref, _ = baca_sheet_otomatis(
            uploaded_file,
            SHEET_REF,
            [
                "Nama Pelatihan",
                "Nama_pelatihan",
                "Silabus",
                "Unit Kompetensi",
                "Unit_Kompetensi",
                "Level Kompetensi",
                "Level_Kompetensi",
            ],
        )
        meta["catatan"].append(
            f"Sheet ref ditemukan dan dipakai sebagai sumber utama ML. Jumlah baris ref: {len(df_ref):,}."
        )
    else:
        meta["catatan"].append(
            "Sheet ref tidak ditemukan. Prediksi memakai Histori yang sudah terisi dan Standar Kompetensi sebagai fallback."
        )

    meta["mode"] = "Mengolah Data Pegawai + Standar Kompetensi + Histori Pelatihan + ref jika tersedia"
    meta["header_excel"] = {
        SHEET_DATA_PEGAWAI: header_pegawai,
        SHEET_STANDAR_KOMPETENSI: header_standar,
        SHEET_HISTORI_PELATIHAN: header_histori,
    }
    if header_ref is not None:
        meta["header_excel"][SHEET_REF] = header_ref

    # ------------------------------------------------------------
    # Validasi kolom wajib.
    # ------------------------------------------------------------
    peg_cols = _cek_col_or_raise(
        df_pegawai,
        SHEET_DATA_PEGAWAI,
        {
            "Nama_Pegawai": ["Nama_Pegawai", "Nama Pegawai"],
            "NIP_Panjang": ["NIP_Panjang", "NIP Panjang", "NIP"],
            "Jabatan": ["Jabatan"],
        },
    )

    std_cols = _cek_col_or_raise(
        df_standar,
        SHEET_STANDAR_KOMPETENSI,
        {
            "Jabatan": ["Jabatan"],
            "Unit_Kompetensi": ["Unit_Kompetensi", "Unit Kompetensi", "Unit_kompetensi"],
            "Level_Kompetensi": ["Level_Kompetensi", "Level Kompetensi", "Level_kompetensi"],
        },
    )

    his_cols = _cek_col_or_raise(
        df_histori,
        SHEET_HISTORI_PELATIHAN,
        {
            "Nama_Pegawai": ["Nama_Pegawai", "Nama Pegawai"],
            "NIP_Panjang": ["NIP_Panjang", "NIP Panjang", "NIP"],
            "Nama_pelatihan": ["Nama_pelatihan", "Nama Pelatihan", "Nama_Pelatihan"],
        },
    )

    peg_nama = peg_cols["Nama_Pegawai"]
    peg_nip = peg_cols["NIP_Panjang"]
    peg_jabatan = peg_cols["Jabatan"]

    std_jabatan = std_cols["Jabatan"]
    std_unit = std_cols["Unit_Kompetensi"]
    std_level = std_cols["Level_Kompetensi"]

    his_nama = his_cols["Nama_Pegawai"]
    his_nip = his_cols["NIP_Panjang"]
    his_pelatihan = his_cols["Nama_pelatihan"]

    his_silabus = find_col(df_histori, ["Silabus"], allow_contains=True)
    his_unit = find_col(df_histori, ["Unit_kompetensi", "Unit Kompetensi", "Unit_Kompetensi"], allow_contains=True)
    his_level = find_col(df_histori, ["Level_kompetensi", "Level Kompetensi", "Level_Kompetensi"], allow_contains=True)

    if his_silabus is None:
        df_histori["Silabus"] = ""
        his_silabus = "Silabus"
        meta["catatan"].append("Kolom Silabus pada Histori Pelatihan tidak ditemukan, dibuat kosong.")

    if his_unit is None:
        df_histori["Unit_kompetensi"] = ""
        his_unit = "Unit_kompetensi"
        meta["catatan"].append("Kolom Unit_kompetensi pada Histori Pelatihan tidak ditemukan, dibuat kosong.")

    if his_level is None:
        df_histori["Level_kompetensi"] = ""
        his_level = "Level_kompetensi"
        meta["catatan"].append("Kolom Level_kompetensi pada Histori Pelatihan tidak ditemukan, dibuat kosong.")

    # ------------------------------------------------------------
    # Bersihkan kolom kunci.
    # ------------------------------------------------------------
    force_text_key(df_pegawai, peg_nip)
    force_text_key(df_histori, his_nip)

    for df_tmp, cols in [
        (df_pegawai, [peg_nama, peg_nip, peg_jabatan]),
        (df_standar, [std_jabatan, std_unit]),
        (df_histori, [his_nama, his_nip, his_pelatihan, his_silabus, his_unit]),
    ]:
        for c in cols:
            _strip_text_column(df_tmp, c)

    # ------------------------------------------------------------
    # Diagnostik data awal sebelum prediksi/default level.
    # ------------------------------------------------------------
    diagnostik_awal = buat_diagnostik_excel(
        df_pegawai=df_pegawai,
        df_standar=df_standar,
        df_histori=df_histori,
        df_ref=df_ref,
    )
    meta["diagnostik_data"] = _diagnostik_ringkas(diagnostik_awal)

    # ------------------------------------------------------------
    # Prediksi ML.
    # ------------------------------------------------------------
    pred_meta: Dict[str, Any] = {}
    if aktifkan_prediksi:
        df_histori, pred_meta = prediksi_histori_unit_level(
            df_histori=df_histori,
            df_standar=df_standar,
            df_ref=df_ref,
            batas_confidence=batas_confidence,
            metode_prediksi=metode_prediksi,
        )

        meta["catatan"].append(pred_meta.get("pesan", "Prediksi selesai."))
        meta["metode_prediksi"] = pred_meta.get("metode_prediksi", metode_prediksi)

        # Refresh kolom setelah prediksi karena modul prediksi bisa membuat kolom baru.
        his_unit = find_col(df_histori, ["Unit_kompetensi", "Unit Kompetensi", "Unit_Kompetensi"], allow_contains=True)
        his_level = find_col(df_histori, ["Level_kompetensi", "Level Kompetensi", "Level_Kompetensi"], allow_contains=True)

    else:
        pred_meta = {
            "pesan": "Prediksi ML tidak diaktifkan.",
            "metode_prediksi": "Tidak aktif",
            "jumlah_unit_diisi": 0,
            "jumlah_level_diisi": 0,
            "jumlah_perlu_review": 0,
            "jumlah_confidence_rendah": 0,
            "unit_kosong_awal": int(df_histori[his_unit].apply(is_empty_value).sum()) if his_unit else 0,
            "level_kosong_awal": int(df_histori[his_level].apply(is_empty_value).sum()) if his_level else 0,
            "unit_kosong_akhir": int(df_histori[his_unit].apply(is_empty_value).sum()) if his_unit else 0,
            "level_kosong_akhir": int(df_histori[his_level].apply(is_empty_value).sum()) if his_level else 0,
        }
        meta["catatan"].append("Prediksi ML tidak diaktifkan dari sidebar.")
        meta["metode_prediksi"] = "Tidak aktif"

        # Kolom audit minimal agar tabel prediksi tetap aman.
        if "Status_Prediksi" not in df_histori.columns:
            df_histori["Status_Prediksi"] = "Prediksi tidak aktif"
        if "Confidence_Prediksi" not in df_histori.columns:
            df_histori["Confidence_Prediksi"] = 0.0

# ------------------------------------------------------------
# Mode praktis: isi level kosong dengan default.
# ------------------------------------------------------------
jumlah_level_default = 0

if isi_level_default:

# Jika kolom level belum ada
if his_level is None:
    df_histori["Level_kompetensi"] = np.nan
    his_level = "Level_kompetensi"

# Pastikan kolom numerik
df_histori[his_level] = pd.to_numeric(
    df_histori[his_level],
    errors="coerce"
)

# Cari level kosong
mask_level_kosong = df_histori[his_level].isna()

jumlah_level_default = int(mask_level_kosong.sum())

if jumlah_level_default > 0:

    nilai_default = float(nilai_level_default)

    # Isi level kompetensi kosong
    df_histori.loc[
        mask_level_kosong,
        his_level
    ] = nilai_default

    # Sinkronkan prediksi level
    if "Prediksi_Level_Kompetensi" in df_histori.columns:

        df_histori["Prediksi_Level_Kompetensi"] = pd.to_numeric(
            df_histori["Prediksi_Level_Kompetensi"],
            errors="coerce"
        )

        df_histori.loc[
            mask_level_kosong,
            "Prediksi_Level_Kompetensi"
        ] = nilai_default

    # Tambahkan status prediksi
    if "Status_Prediksi" in df_histori.columns:

        status_awal = (
            df_histori.loc[
                mask_level_kosong,
                "Status_Prediksi"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        status_baru = (
            status_awal
            + f" | Level default {nilai_level_default}"
        )

        status_baru = (
            status_baru
            .str.replace(r"^\s*\|\s*", "", regex=True)
            .str.replace(r"\s+\|\s+\|", " | ", regex=True)
            .str.strip()
        )

        df_histori.loc[
            mask_level_kosong,
            "Status_Prediksi"
        ] = status_baru

        # Logging metadata
        meta["catatan"].append(
            f"Mode praktis aktif: "
            f"{jumlah_level_default:,} "
            f"Level Kompetensi kosong "
            f"diisi default {nilai_level_default}."
        )

    else:

        meta["catatan"].append(
            "Mode praktis aktif, tetapi tidak ada "
            "Level Kompetensi kosong yang perlu diisi."
        )
    # ------------------------------------------------------------
    # Simpan summary prediksi dan dataframe prediksi.
    # ------------------------------------------------------------
    pred_meta["jumlah_level_default"] = jumlah_level_default
    if his_level is not None and his_level in df_histori.columns:
        pred_meta["level_kosong_akhir"] = int(df_histori[his_level].apply(is_empty_value).sum())
    else:
        pred_meta["level_kosong_akhir"] = 0

    if his_unit is not None and his_unit in df_histori.columns:
        pred_meta["unit_kosong_akhir"] = int(df_histori[his_unit].apply(is_empty_value).sum())
    else:
        pred_meta["unit_kosong_akhir"] = 0

    meta["prediksi_summary"] = pred_meta
    meta["df_prediksi_ml"] = df_histori.copy()

    # ------------------------------------------------------------
    # Siapkan data numerik untuk evaluasi.
    # ------------------------------------------------------------
    df_standar[std_level] = pd.to_numeric(df_standar[std_level], errors="coerce").fillna(0)
    df_histori[his_level] = pd.to_numeric(df_histori[his_level], errors="coerce").fillna(0)

    # Pastikan kolom unit tidak kosong agar groupby aman.
    df_histori[his_unit] = df_histori[his_unit].fillna("").astype(str).str.strip()

    # ------------------------------------------------------------
    # Ambil level pelatihan tertinggi per pegawai dan unit kompetensi.
    # ------------------------------------------------------------
    histori_level = (
        df_histori
        .groupby([his_nama, his_nip, his_unit], as_index=False)
        .agg({
            his_level: "max",
            his_pelatihan: gabung_unik,
        })
    )

    # Rekomendasi pelatihan per unit kompetensi.
    rekomendasi_unit = (
        df_histori
        .groupby(his_unit, as_index=False)[his_pelatihan]
        .apply(gabung_unik)
        .rename(columns={his_unit: std_unit, his_pelatihan: "Rekomendasi_Unit"})
    )

    # ------------------------------------------------------------
    # Evaluasi: pegawai x standar jabatan x histori.
    # ------------------------------------------------------------
    detail = df_pegawai.merge(
        df_standar,
        left_on=peg_jabatan,
        right_on=std_jabatan,
        how="left",
        suffixes=("", "_Standar"),
    )

    detail = detail.merge(
        histori_level,
        left_on=[peg_nama, peg_nip, std_unit],
        right_on=[his_nama, his_nip, his_unit],
        how="left",
        suffixes=("", "_Histori"),
    )

    detail = detail.merge(rekomendasi_unit, on=std_unit, how="left")

    detail[his_level] = detail[his_level].fillna(0)
    detail[his_pelatihan] = detail[his_pelatihan].fillna("-")
    detail["Rekomendasi_Unit"] = detail["Rekomendasi_Unit"].fillna("Belum ada rekomendasi pelatihan")

    detail["Cocok"] = detail[his_level] >= detail[std_level]
    detail["Gap_Level"] = (detail[std_level] - detail[his_level]).clip(lower=0)

    detail["Kompetensi_Item"] = detail.apply(
        lambda r: f"{r[std_unit]} lv {int(r[std_level])}" if pd.notna(r.get(std_unit, None)) and str(r.get(std_unit, "")).strip() else "-",
        axis=1,
    )

    detail["Kompetensi_Cocok_Item"] = detail.apply(
        lambda r: f"{r[std_unit]} lv {int(r[std_level])}" if bool(r["Cocok"]) and pd.notna(r.get(std_unit, None)) and str(r.get(std_unit, "")).strip() else "-",
        axis=1,
    )

    detail["Kompetensi_Kurang_Item"] = detail.apply(
        lambda r: (
            f"{r[std_unit]} (butuh level {int(r[std_level])}, saat ini {int(r[his_level])})"
            if (not bool(r["Cocok"])) and pd.notna(r.get(std_unit, None)) and str(r.get(std_unit, "")).strip()
            else "-"
        ),
        axis=1,
    )

    detail["Rekomendasi_Item"] = detail.apply(
        lambda r: r["Rekomendasi_Unit"] if not bool(r["Cocok"]) else "-",
        axis=1,
    )

    # ------------------------------------------------------------
    # Agregasi hasil per pegawai.
    # ------------------------------------------------------------
    group_cols = [peg_nama, peg_nip, peg_jabatan]

    optional_map = {}
    for wanted in ["Unit_Es_IV", "Unit_Es_III", "Unit_Es_II"]:
        found = find_col(df_pegawai, [wanted, wanted.replace("_", " ")], allow_contains=True)
        if found and found not in group_cols:
            group_cols.append(found)
            optional_map[found] = wanted

    hasil = (
        detail
        .groupby(group_cols, as_index=False)
        .agg({
            std_unit: "count",
            "Cocok": "sum",
            "Kompetensi_Item": gabung_unik,
            "Kompetensi_Cocok_Item": gabung_unik,
            "Kompetensi_Kurang_Item": gabung_unik,
            "Rekomendasi_Item": gabung_unik,
        })
    )

    hasil = hasil.rename(columns={
        peg_nama: "Nama_Pegawai",
        peg_nip: "NIP_Panjang",
        peg_jabatan: "Jabatan",
        std_unit: "Jumlah_Kompetensi_Diuji",
        "Cocok": "Jumlah_Kompetensi_Cocok",
        "Kompetensi_Item": "Daftar_Kompetensi",
        "Kompetensi_Cocok_Item": "Kompetensi_Cocok",
        "Kompetensi_Kurang_Item": "Kompetensi_Kurang",
        "Rekomendasi_Item": "Rekomendasi_Pelatihan",
    })

    hasil = hasil.rename(columns=optional_map)

    hasil["Skor_Kecocokan_%"] = (
        hasil["Jumlah_Kompetensi_Cocok"] / hasil["Jumlah_Kompetensi_Diuji"] * 100
    ).fillna(0).round(2)

    hasil["Status_Kompetensi"] = hasil["Skor_Kecocokan_%"].apply(
        lambda x: "Kompeten" if x >= 100 else "Tidak Kompeten"
    )

    hasil["Jabatan_Diuji"] = hasil["Jabatan"]

    kolom_akhir = [
        "Nama_Pegawai",
        "NIP_Panjang",
        "Jabatan",
        "Unit_Es_IV",
        "Unit_Es_III",
        "Unit_Es_II",
        "Jabatan_Diuji",
        "Skor_Kecocokan_%",
        "Status_Kompetensi",
        "Daftar_Kompetensi",
        "Kompetensi_Cocok",
        "Kompetensi_Kurang",
        "Rekomendasi_Pelatihan",
        "Jumlah_Kompetensi_Diuji",
        "Jumlah_Kompetensi_Cocok",
    ]

    hasil = hasil[[c for c in kolom_akhir if c in hasil.columns]].copy()

    return hasil, meta
