# app_2.2.py
# Dashboard Evaluasi Kompetensi Pegawai - versi 2.2
# Jalankan dengan:
# streamlit run app_2.2.py
#
# Catatan desain versi 2.2:
# - Mempertahankan tampilan dan alur utama app_2.0 yang sudah bagus untuk demo.
# - Menambahkan cache proses Excel agar tidak diproses ulang setiap interaksi.
# - Membatasi tampilan tabel agar aplikasi tetap cepat pada data besar.
# - Menambahkan diagnostik kualitas data agar hasil evaluasi lebih mudah dipercaya.
# - Memakai modul versi 2.2 dengan nama sederhana dan informatif.

from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd
import streamlit as st

from modul_excel_22 import normalisasi_kolom
from modul_evaluasi_22 import convert_df_to_excel, hitung_top_item, proses_file_upload
from modul_tampilan_22 import render_top_list, tampilkan_detail_pegawai


# ============================================================
# Konfigurasi halaman
# ============================================================
st.set_page_config(
    page_title="SIKAP v2.2",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# CSS dashboard
# Sebagian besar gaya dipertahankan dari app_2.0 karena tampilannya
# ============================================================
st.markdown(
    """
<style>
.main-title {font-size: 40px; font-weight: 800; color: #F8FAFC; margin-bottom: 0px;}
.subtitle {font-size: 16px; color: #CBD5E1; margin-bottom: 24px;}
.section-title {font-size: 24px; font-weight: 800; margin-top: 22px; margin-bottom: 12px; color: #F8FAFC;}
.info-box {background-color: #0F172A; padding: 18px; border-left: 5px solid #38BDF8; border-radius: 10px; color: #E5E7EB;}
.metric-card {background: linear-gradient(135deg, #1E293B, #0F172A); padding: 20px; border-radius: 18px; border: 1px solid #334155; text-align: center;}
.metric-number {font-size: 32px; font-weight: 800; color: #38BDF8;}
.metric-label {font-size: 14px; color: #CBD5E1;}
.profile-card {background: #0F172A; border: 1px solid #334155; border-radius: 20px; padding: 24px; margin-top: 12px; margin-bottom: 18px;}
.profile-layout {display: grid; grid-template-columns: 140px 1fr; gap: 24px; align-items: center;}
.profile-avatar {width: 118px; height: 118px; border-radius: 999px; background: linear-gradient(135deg, #2563EB, #0F172A); border: 2px solid #60A5FA; display: flex; align-items: center; justify-content: center; font-size: 64px; box-shadow: 0 18px 35px rgba(37, 99, 235, 0.25);}
.profile-title {font-size: 24px; font-weight: 800; color: #F8FAFC; margin-bottom: 16px;}
.profile-row {font-size: 16px; color: #E5E7EB; margin-bottom: 7px;}
.profile-label {color: #93C5FD; font-weight: 700; display: inline-block; min-width: 82px;}
.competency-box {background: #111827; border: 1px solid #475569; border-radius: 18px; padding: 18px 20px; min-height: 260px;}
.competency-title {font-size: 20px; font-weight: 800; color: #F8FAFC; margin-bottom: 14px;}
.competency-item {padding: 10px 12px; border-radius: 12px; margin-bottom: 10px; color: #E5E7EB; border: 1px solid #334155;}
.competency-ok {background: linear-gradient(135deg, #1E3A8A, #0F172A); border: 1px solid #60A5FA; color: #DBEAFE;}
.competency-gap {background: linear-gradient(135deg, #7F1D1D, #450A0A); border: 1px solid #F87171; color: #FEE2E2;}
.competency-neutral {background: #1E293B; border: 1px solid #334155; color: #E5E7EB;}
.status-pill {display: inline-block; margin-left: 8px; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 700;}
.status-pill-ok {background: #1D4ED8; color: #DBEAFE;}
.status-pill-gap {background: #991B1B; color: #FEE2E2;}
.status-good {background: linear-gradient(135deg, #064E3B, #022C22); border: 1px solid #10B981; border-radius: 18px; padding: 28px; text-align: center; color: #D1FAE5; font-size: 25px; font-weight: 900; min-height: 260px; display: flex; align-items: center; justify-content: center;}
.status-bad {background: linear-gradient(135deg, #7F1D1D, #450A0A); border: 1px solid #F87171; border-radius: 18px; padding: 18px 20px; color: #FEE2E2; min-height: 260px;}
.rekom-title {font-size: 20px; font-weight: 800; color: #FEE2E2; margin-bottom: 12px;}
.rekom-item {margin-bottom: 9px; font-size: 15px; line-height: 1.35;}
.small-note {color: #94A3B8; font-size: 13px;}
.warning-card {background: #422006; color: #FEF3C7; border: 1px solid #F59E0B; border-radius: 14px; padding: 14px 16px; margin-bottom: 14px;}
.insight-card {background: #0F172A; border: 1px solid #334155; border-radius: 18px; padding: 18px 20px; min-height: 150px;}
.insight-title {font-size: 18px; font-weight: 800; color: #F8FAFC; margin-bottom: 10px;}
.insight-text {font-size: 15px; line-height: 1.5; color: #E5E7EB;}
.presentation-note {background: #172554; color: #DBEAFE; border: 1px solid #3B82F6; border-radius: 16px; padding: 16px 18px; margin-top: 12px; margin-bottom: 14px; line-height: 1.5;}
.top-list-item {padding: 8px 10px; margin-bottom: 7px; border-radius: 10px; background: #111827; border: 1px solid #334155; color: #E5E7EB;}
.method-card {background: #0F172A; border: 1px solid #334155; border-radius: 16px; padding: 16px 18px; margin-bottom: 12px; color: #E5E7EB; line-height: 1.55;}
.method-card b {color: #93C5FD;}
.method-step {background: #111827; border: 1px solid #334155; border-radius: 14px; padding: 12px 14px; margin-bottom: 8px; color: #E5E7EB;}
.method-warning {background: #422006; color: #FEF3C7; border: 1px solid #F59E0B; border-radius: 14px; padding: 14px 16px; margin-top: 10px; margin-bottom: 10px; line-height: 1.55;}
.data-quality-card {background: #0B1220; border: 1px solid #334155; border-radius: 16px; padding: 14px 16px; margin-bottom: 10px; color: #E5E7EB;}
.data-quality-title {font-size: 16px; font-weight: 800; color: #93C5FD; margin-bottom: 8px;}
.data-quality-row {font-size: 14px; margin-bottom: 4px; color: #CBD5E1;}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Helper kecil untuk app utama
# ============================================================
def pilihan_jumlah_ke_int(pilihan: str, total: int) -> int:
    """Mengubah pilihan jumlah baris dari UI menjadi angka."""
    if pilihan == "All":
        return int(total)
    return int(pilihan)


def batasi_dataframe(df: pd.DataFrame, pilihan: str) -> pd.DataFrame:
    """Membatasi dataframe sesuai pilihan 10/50/100/All."""
    if df is None:
        return pd.DataFrame()
    n = pilihan_jumlah_ke_int(pilihan, len(df))
    return df.head(n).copy()


def format_int(x: Any) -> str:
    """Format angka agar mudah dibaca."""
    try:
        return f"{int(x):,}".replace(",", ".")
    except Exception:
        return str(x)


def render_diagnostik(meta: dict) -> None:
    """Menampilkan ringkasan kualitas data dari metadata proses."""
    diagnostik = meta.get("diagnostik_data", {}) if isinstance(meta, dict) else {}
    if not diagnostik:
        st.info("Diagnostik data belum tersedia dari modul evaluasi.")
        return

    cols = st.columns(4)
    daftar_sheet = [
        ("Data Pegawai", "pegawai"),
        ("Standar Kompetensi", "standar"),
        ("Histori Pelatihan", "histori"),
        ("Ref", "ref"),
    ]

    for i, (judul, key) in enumerate(daftar_sheet):
        item = diagnostik.get(key, {}) or {}
        with cols[i % 4]:
            rows = ""
            for nama, nilai in item.items():
                rows += f'<div class="data-quality-row"><b>{nama}</b>: {format_int(nilai)}</div>'
            if not rows:
                rows = '<div class="data-quality-row">Tidak ada data.</div>'
            st.markdown(
                f"""
                <div class="data-quality-card">
                    <div class="data-quality-title">{judul}</div>
                    {rows}
                </div>
                """,
                unsafe_allow_html=True,
            )


@st.cache_data(show_spinner=False)
def proses_excel_cached(
    file_bytes: bytes,
    aktifkan_prediksi: bool,
    batas_confidence: int,
    isi_level_default: bool,
    nilai_level_default: int,
    metode_prediksi: str,
):
    """Cache proses Excel.

    Streamlit lebih stabil mencache bytes daripada object uploaded_file.
    Jika salah satu parameter berubah, cache akan dihitung ulang.
    """
    return proses_file_upload(
        BytesIO(file_bytes),
        aktifkan_prediksi=aktifkan_prediksi,
        batas_confidence=batas_confidence,
        isi_level_default=isi_level_default,
        nilai_level_default=nilai_level_default,
        metode_prediksi=metode_prediksi,
    )


# ============================================================
# Header aplikasi
# ============================================================
st.markdown('<div class="main-title">📊 SIKAP: Sistem Informasi Kompetensi dan Analisis Pelatihan</div>', unsafe_allow_html=True)
  
# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader("Upload file Excel utama", type=["xlsx", "xls"])

    st.divider()
    st.header("🤖 Prediksi ML")
    aktifkan_prediksi = st.checkbox(
        "Aktifkan prediksi Unit/Level kosong",
        value=True,
        help="Matikan jika Unit_kompetensi dan Level_kompetensi pada Histori Pelatihan sudah lengkap.",
    )
    metode_prediksi = st.selectbox(
        "Metode prediksi",
        options=["SBERT", "TF-IDF"],
        index=0,
        help="SBERT lebih memahami makna, tetapi lebih berat. TF-IDF lebih cepat namun kurang akurat.",
    )
    batas_prediksi = st.slider(
        "Batas minimal confidence untuk DIISI (%)",
        min_value=40,
        max_value=95,
        value=75,
        step=1,
    )

    st.divider()
    st.header("⚡ Mode Praktis")
    isi_level_default = st.checkbox(
        "Isi Level Kompetensi kosong dengan default",
        value=True,
        help="Untuk prototype/demo. Jika dipakai resmi, data level sebaiknya dilengkapi dari sumber valid.",
    )
    nilai_level_default = st.selectbox(
        "Nilai default Level Kompetensi",
        options=[1, 2, 3, 4, 5],
        index=0,
    )

    st.divider()
    st.header("📋 Tampilan Tabel")
    jumlah_baris_prediksi = st.selectbox(
        "Jumlah baris tabel prediksi ML",
        options=["10", "50", "100", "All"],
        index=0,
    )
    jumlah_baris_pegawai = st.selectbox(
        "Jumlah pegawai pada tabel list",
        options=["10", "50", "100", "All"],
        index=0,
    )

    st.divider()
    st.header("ℹ️ Petunjuk")
    st.write(
        """
        File Excel disarankan punya sheet:
        - `Data Pegawai`
        - `Standar Kompetensi`
        - `Histori Pelatihan`
        - `ref`

        Pada versi 2.2, sheet `ref` menjadi sumber utama prediksi ML.
        Kolom `Silabus` ikut dipakai untuk membandingkan kemiripan pelatihan.

        `by: Kelompok II` 
         PJJ Data analytics angkatan II 2026 
        """
    )


# ============================================================
# Kondisi awal: belum upload file
# ============================================================
if uploaded_file is None:
    st.markdown(
        """
        <div class="info-box">
        Silakan upload file Excel. Setelah diproses, dashboard akan menampilkan ringkasan evaluasi,
        prediksi ML, diagnostik kualitas data, pencarian pegawai, detail kompetensi, dan file unduhan.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ============================================================
# Proses file
# ============================================================
try:
    file_bytes = uploaded_file.getvalue()

    with st.spinner("Membaca dan memproses file Excel..."):
        df, meta = proses_excel_cached(
            file_bytes=file_bytes,
            aktifkan_prediksi=aktifkan_prediksi,
            batas_confidence=batas_prediksi,
            isi_level_default=isi_level_default,
            nilai_level_default=nilai_level_default,
            metode_prediksi=metode_prediksi,
        )

    df = normalisasi_kolom(df)

    st.success("✅ File berhasil dibaca dan diproses!")

    # ------------------------------------------------------------
    # Cek teknis pembacaan file
    # ------------------------------------------------------------
    with st.expander("Cek teknis pembacaan file yang telah diproses"):
        st.write("Mode:", meta.get("mode"))
        st.write("Metode prediksi:", meta.get("metode_prediksi"))
        st.write("Sheet tersedia:", meta.get("sheet_names"))
        if "header_excel" in meta:
            st.write("Header yang dipakai:", meta.get("header_excel"))
        for catatan in meta.get("catatan", []):
            st.write("-", catatan)
        st.write("Kolom hasil:", list(df.columns))

    # ------------------------------------------------------------
    # Ringkasan Prediksi ML
    # ------------------------------------------------------------
    if meta.get("prediksi_summary"):
        ps = meta.get("prediksi_summary", {})
        st.markdown('<div class="section-title">🤖 Ringkasan Prediksi ML</div>', unsafe_allow_html=True)
        ml1, ml2, ml3, ml4, ml5, ml6 = st.columns(6)
        ml1.metric("Unit diisi", ps.get("jumlah_unit_diisi", 0))
        ml2.metric("Level diisi", ps.get("jumlah_level_diisi", 0))
        ml3.metric("Level default", ps.get("jumlah_level_default", 0))
        ml4.metric("Perlu review", ps.get("jumlah_perlu_review", 0))
        ml5.metric("Conf. rendah", ps.get("jumlah_confidence_rendah", 0))
        ml6.metric("Level kosong akhir", ps.get("level_kosong_akhir", 0))

        if ps.get("jumlah_level_default", 0) > 0:
            st.info(
                f"Mode praktis aktif: {ps.get('jumlah_level_default', 0):,} baris Level Kompetensi kosong "
                "diisi default. Gunakan sebagai asumsi prototype dan validasi ulang bila dipakai resmi."
            )

        df_pred = meta.get("df_prediksi_ml")
        if df_pred is not None:
            with st.expander("Lihat tabel hasil prediksi / perlu review"):
                kolom_prediksi = [
                    "Nama_Pegawai", "NIP_Panjang", "Nama_pelatihan", "Silabus",
                    "Unit_Asli", "Level_Asli",
                    "Prediksi_Unit_Kompetensi", "Prediksi_Level_Kompetensi",
                    "Nama_Pelatihan_Termirip", "Confidence_Prediksi",
                    "Sumber_Prediksi", "Metode_Prediksi", "Status_Prediksi",
                ]
                kolom_prediksi = [c for c in kolom_prediksi if c in df_pred.columns]

                status_opsi_ml = ["Semua"]
                if "Status_Prediksi" in df_pred.columns:
                    status_opsi_ml += sorted(df_pred["Status_Prediksi"].dropna().astype(str).unique().tolist())
                filter_ml = st.selectbox("Filter status prediksi", status_opsi_ml, key="filter_status_prediksi_app22")

                df_show_ml_full = df_pred.copy()
                if filter_ml != "Semua" and "Status_Prediksi" in df_show_ml_full.columns:
                    df_show_ml_full = df_show_ml_full[df_show_ml_full["Status_Prediksi"].astype(str) == filter_ml]

                df_show_ml = batasi_dataframe(df_show_ml_full.reset_index(drop=True), jumlah_baris_prediksi)
                st.caption(
                    f"Menampilkan {format_int(len(df_show_ml))} dari {format_int(len(df_show_ml_full))} baris prediksi. "
                    "Gunakan filter status untuk mempersempit data."
                )
                st.dataframe(df_show_ml[kolom_prediksi], use_container_width=True, height=320, hide_index=True)

                st.download_button(
                    "Download hasil prediksi",
                    data=convert_df_to_excel(df_show_ml_full[kolom_prediksi]),
                    file_name="hasil_prediksi_app_2_2.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    # ------------------------------------------------------------
    # Diagnostik kualitas data
    # ------------------------------------------------------------
    with st.expander("🧪 Diagnostik kualitas data", expanded=False):
        st.caption("Gunakan bagian ini untuk melihat potensi masalah data sebelum hasil evaluasi dipakai resmi.")
        render_diagnostik(meta)

    # ------------------------------------------------------------
    # Pencarian dan filter pegawai
    # ------------------------------------------------------------
    st.markdown('<div class="section-title">🔎 Pencarian Pegawai</div>', unsafe_allow_html=True)
    col_cari1, col_cari2, col_cari3 = st.columns([2, 1, 1])

    with col_cari1:
        kata_kunci = st.text_input(
            "Cari berdasarkan nama atau NIP",
            placeholder="Contoh: Pegawai001 atau 198001012010011001",
        )

    with col_cari2:
        status_opsi = ["Semua"]
        if "Status_Kompetensi" in df.columns:
            status_opsi += sorted(df["Status_Kompetensi"].dropna().astype(str).unique().tolist())
        status_filter = st.selectbox("Status", status_opsi)

    with col_cari3:
        jabatan_opsi = ["Semua"]
        if "Jabatan" in df.columns:
            jabatan_opsi += sorted(df["Jabatan"].dropna().astype(str).unique().tolist())
        jabatan_filter = st.selectbox("Jabatan", jabatan_opsi)

    df_filtered = df.copy()

    if status_filter != "Semua" and "Status_Kompetensi" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Status_Kompetensi"].astype(str) == status_filter]

    if jabatan_filter != "Semua" and "Jabatan" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Jabatan"].astype(str) == jabatan_filter]

    if kata_kunci:
        if "Nama_Pegawai" in df_filtered.columns:
            mask_nama = df_filtered["Nama_Pegawai"].astype(str).str.contains(kata_kunci, case=False, na=False)
        else:
            mask_nama = pd.Series(False, index=df_filtered.index)

        if "NIP_Panjang" in df_filtered.columns:
            mask_nip = df_filtered["NIP_Panjang"].astype(str).str.contains(kata_kunci, case=False, na=False)
        else:
            mask_nip = pd.Series(False, index=df_filtered.index)

        df_filtered = df_filtered[mask_nama | mask_nip]

    # ------------------------------------------------------------
    # Ringkasan Evaluasi
    # ------------------------------------------------------------
    st.markdown('<div class="section-title">📌 Ringkasan Evaluasi</div>', unsafe_allow_html=True)

    total_pegawai = len(df_filtered)
    kompeten = int((df_filtered["Status_Kompetensi"] == "Kompeten").sum()) if "Status_Kompetensi" in df_filtered.columns else 0
    tidak_kompeten = int((df_filtered["Status_Kompetensi"] == "Tidak Kompeten").sum()) if "Status_Kompetensi" in df_filtered.columns else 0
    rata_skor = 0
    if "Skor_Kecocokan_%" in df_filtered.columns:
        rata_skor = round(pd.to_numeric(df_filtered["Skor_Kecocokan_%"], errors="coerce").mean(), 2)
        if pd.isna(rata_skor):
            rata_skor = 0

    persen_kompeten = round((kompeten / total_pegawai * 100), 2) if total_pegawai else 0
    persen_tidak_kompeten = round((tidak_kompeten / total_pegawai * 100), 2) if total_pegawai else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.markdown(f'<div class="metric-card"><div class="metric-number">{format_int(total_pegawai)}</div><div class="metric-label">Total Data</div></div>', unsafe_allow_html=True)
    kpi2.markdown(f'<div class="metric-card"><div class="metric-number">{format_int(kompeten)}</div><div class="metric-label">Kompeten</div></div>', unsafe_allow_html=True)
    kpi3.markdown(f'<div class="metric-card"><div class="metric-number">{format_int(tidak_kompeten)}</div><div class="metric-label">Tidak Kompeten</div></div>', unsafe_allow_html=True)
    kpi4.markdown(f'<div class="metric-card"><div class="metric-number">{rata_skor}%</div><div class="metric-label">Rata-rata Skor</div></div>', unsafe_allow_html=True)

    # ------------------------------------------------------------
    # Insight presentasi
    # ------------------------------------------------------------
    st.markdown('<div class="section-title">📊 Insight untuk Presentasi</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="presentation-note">
        <b>Ringkasan narasi:</b> Dari <b>{format_int(total_pegawai)}</b> pegawai yang dianalisis,
        terdapat <b>{format_int(kompeten)}</b> pegawai kompeten (<b>{persen_kompeten}%</b>) dan
        <b>{format_int(tidak_kompeten)}</b> pegawai belum kompeten (<b>{persen_tidak_kompeten}%</b>).
        Rata-rata skor kecocokan adalah <b>{rata_skor}%</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    insight_col1, insight_col2 = st.columns(2)
    top_gap = hitung_top_item(df_filtered, "Kompetensi_Kurang", top_n=5, bersihkan_gap=True)
    top_rekom = hitung_top_item(df_filtered, "Rekomendasi_Pelatihan", top_n=5, bersihkan_gap=False)

    with insight_col1:
        render_top_list("Top 5 Gap Kompetensi", top_gap, "Belum ada gap kompetensi pada filter ini.")
    with insight_col2:
        render_top_list("Top 5 Rekomendasi Pelatihan", top_rekom, "Belum ada rekomendasi pelatihan pada filter ini.")

    # ------------------------------------------------------------
    # Metodologi dan asumsi
    # ------------------------------------------------------------
    with st.expander("Metodologi & asumsi prototype", expanded=False):
        ps = meta.get("prediksi_summary", {})
        st.markdown(
            """
            <div class="method-card">
            <b>Tujuan aplikasi:</b><br>
            Aplikasi ini memetakan histori pelatihan pegawai terhadap standar kompetensi jabatan untuk menghasilkan status kompeten/tidak kompeten, gap kompetensi, dan rekomendasi pelatihan.
            </div>
            """,
            unsafe_allow_html=True,
        )
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                """
                <div class="method-step"><b>1. Input data</b><br>File Excel dibaca dari sheet Data Pegawai, Standar Kompetensi, Histori Pelatihan, dan ref jika tersedia.</div>
                <div class="method-step"><b>2. Pembersihan data</b><br>Nama kolom dirapikan, NIP disamakan sebagai teks, dan data kosong dikenali secara konsisten.</div>
                <div class="method-step"><b>3. Prediksi</b><br>Nama pelatihan dan silabus dibandingkan dengan sheet ref menggunakan SBERT atau TF-IDF.</div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                """
                <div class="method-step"><b>4. Sheet ref</b><br>Sheet ref menjadi sumber utama ML. Histori yang sudah terisi dan standar kompetensi menjadi referensi tambahan.</div>
                <div class="method-step"><b>5. Evaluasi kompetensi</b><br>Kompetensi sesuai jika unit cocok dengan standar jabatan dan level pelatihan memenuhi level minimal.</div>
                <div class="method-step"><b>6. Output</b><br>Aplikasi menampilkan dashboard, detail pegawai, insight presentasi, diagnostik data, dan file unduhan.</div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            f"""
            <div class="method-warning">
            <b>Catatan asumsi prototype:</b><br>
            Level default digunakan pada <b>{format_int(ps.get('jumlah_level_default', 0))}</b> baris bila mode praktis aktif.
            Pada prototype ini, pegawai dinyatakan kompeten jika seluruh kompetensi jabatan terpenuhi.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------
    # Tabel list pegawai
    # ------------------------------------------------------------
    st.markdown('<div class="section-title">📋 Tabel List Pegawai</div>', unsafe_allow_html=True)
    st.caption("Klik salah satu baris pada tabel untuk menampilkan detail pegawai.")

    kolom_tampil = [
        "Nama_Pegawai", "NIP_Panjang", "Jabatan", "Unit_Es_IV", "Unit_Es_III", "Unit_Es_II",
        "Skor_Kecocokan_%", "Status_Kompetensi", "Daftar_Kompetensi",
        "Kompetensi_Kurang", "Rekomendasi_Pelatihan",
    ]
    kolom_tampil = [c for c in kolom_tampil if c in df_filtered.columns]

    df_tabel_full = df_filtered.reset_index(drop=True)
    df_tabel_show = batasi_dataframe(df_tabel_full, jumlah_baris_pegawai)

    st.caption(
        f"Menampilkan {format_int(len(df_tabel_show))} dari {format_int(len(df_tabel_full))} pegawai. "
        "Gunakan pencarian/filter untuk mempersempit data."
    )

    event_tabel = st.dataframe(
        df_tabel_show[kolom_tampil],
        use_container_width=True,
        height=360,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="tabel_list_pegawai_app22",
    )

    try:
        baris_dipilih = event_tabel.selection.rows
    except Exception:
        baris_dipilih = []

    row_untuk_detail = None
    if baris_dipilih:
        # Penting: ambil dari df_tabel_show agar klik sesuai data yang sedang tampil.
        row_untuk_detail = df_tabel_show.iloc[baris_dipilih[0]]
    elif kata_kunci:
        if len(df_tabel_full) == 0:
            st.warning("Data pegawai tidak ditemukan. Coba masukkan nama atau NIP lain.")
        elif len(df_tabel_full) == 1:
            row_untuk_detail = df_tabel_full.iloc[0]
        else:
            st.info("Ditemukan lebih dari satu pegawai. Klik salah satu baris pada tabel untuk melihat detail pegawai.")

    if row_untuk_detail is not None:
        tampilkan_detail_pegawai(row_untuk_detail)
    elif not kata_kunci:
        st.info("Masukkan nama/NIP atau klik salah satu baris pada tabel agar detail pegawai tampil.")

    # ------------------------------------------------------------
    # Download data
    # ------------------------------------------------------------
    st.markdown('<div class="section-title">📥 Download Data</div>', unsafe_allow_html=True)
    col_download1, col_download2 = st.columns(2)

    with col_download1:
        st.download_button(
            label="Download CSV",
            data=df_filtered.to_csv(index=False).encode("utf-8"),
            file_name="hasil_evaluasi_filtered_app_2_2.csv",
            mime="text/csv",
        )

    with col_download2:
        excel_data = convert_df_to_excel(df_filtered)
        st.download_button(
            label="Download Excel",
            data=excel_data,
            file_name="hasil_evaluasi_filtered_app_2_2.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

except Exception as e:
    st.error("❌ File belum berhasil diproses. Periksa kembali format sheet dan kolom Excel.")
    st.markdown(
        """
        <div class="warning-card">
        Pastikan file Excel memiliki sheet: <b>Data Pegawai</b>, <b>Standar Kompetensi</b>,
        <b>Histori Pelatihan</b>, dan sebaiknya <b>ref</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Detail teknis error"):
        st.exception(e)
