# modul_tampilan_22.py
# ============================================================
# Modul Tampilan Dashboard Evaluasi Kompetensi Pegawai v2.2
# ============================================================
#
# Fungsi utama modul ini:
# 1. Menampilkan top gap kompetensi.
# 2. Menampilkan top rekomendasi pelatihan.
# 3. Menampilkan detail pegawai.
# 4. Membersihkan nilai kosong agar tampilan tidak rusak.
# 5. Mengamankan teks dari Excel agar tidak merusak HTML Streamlit.
#
# Catatan:
# Modul ini tidak melakukan perhitungan evaluasi.
# Perhitungan tetap dilakukan di modul_evaluasi_22.py.

from __future__ import annotations

import html
import math
from typing import Any, List, Tuple

import pandas as pd
import streamlit as st

try:
    from modul_evaluasi_22 import pecah_item, bersihkan_level_item
except Exception:
    # Fallback agar modul tetap bisa dipakai jika import gagal.
    def pecah_item(teks):
        if pd.isna(teks):
            return []
        teks = str(teks).strip()
        if teks == "" or teks == "-" or teks.lower() == "nan":
            return []
        return [x.strip() for x in teks.split(" | ") if x.strip() and x.strip() != "-"]

    def bersihkan_level_item(item):
        item = str(item).strip()
        if "(butuh level" in item:
            nama = item.split("(butuh level")[0].strip()
            try:
                level = item.split("butuh level")[1].split(",")[0].strip()
                return f"{nama} lv {level}"
            except Exception:
                return nama
        return item


# ============================================================
# Helper tampilan aman
# ============================================================

def nilai_aman(value: Any, default: str = "-") -> str:
    """Mengubah nilai apa pun menjadi teks aman untuk ditampilkan."""
    if value is None:
        return default

    try:
        if isinstance(value, float) and math.isnan(value):
            return default
    except Exception:
        pass

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    text = str(value).strip()

    if text == "" or text.lower() in ["nan", "none", "null", "nat"]:
        return default

    return html.escape(text)


def angka_aman(value: Any, default: float = 0.0) -> float:
    """Mengubah nilai menjadi angka float aman."""
    try:
        val = float(value)
        if math.isnan(val):
            return default
        return val
    except Exception:
        return default


def format_skor(value: Any) -> str:
    """Format skor kecocokan agar rapi."""
    val = angka_aman(value, 0.0)
    if val == int(val):
        return f"{int(val)}%"
    return f"{val:.2f}%"


def _filter_item_valid(items: List[str]) -> List[str]:
    hasil = []

    blacklist = {
        "-",
        "nan",
        "none",
        "null",
        "</div>",
        "<div>",
    }

    for item in items:
        text = str(item).strip()

        if not text:
            continue

        if text.lower() in blacklist:
            continue

        # buang tag html nyasar
        if "<div" in text.lower() or "</div" in text.lower() or "div>" in text.lower():
            continue

        hasil.append(text)

    return hasil


# ============================================================
# Komponen top list
# ============================================================

def render_top_list(title: str, df_top, empty_text: str = "Belum ada data."):
    """Menampilkan daftar top item, misalnya top gap atau top rekomendasi."""
    st.markdown(f'<div class="insight-title">{html.escape(str(title))}</div>', unsafe_allow_html=True)

    if df_top is None or len(df_top) == 0:
        st.markdown(
            f'<div class="small-note">{html.escape(str(empty_text))}</div>',
            unsafe_allow_html=True,
        )
        return

    for i, row in df_top.iterrows():
        item = nilai_aman(row.get("Item", "-"))
        jumlah = row.get("Jumlah", 0)

        try:
            jumlah = int(jumlah)
        except Exception:
            jumlah = 0

        st.markdown(
            f"""
            <div class="top-list-item">
                <b>{i + 1}. {item}</b><br>
                <span class="small-note">Muncul pada {jumlah:,} pegawai/baris hasil</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# Komponen detail pegawai
# ============================================================

def _buat_daftar_kompetensi(row) -> List[Tuple[str, str]]:
    """Membuat daftar kompetensi dengan status visual.

    Return list tuple:
    - (nama kompetensi, "ok")
    - (nama kompetensi, "gap")
    - (nama kompetensi, "neutral")
    """
    kompetensi_cocok = [
        bersihkan_level_item(x)
        for x in pecah_item(row.get("Kompetensi_Cocok", ""))
    ]

    kompetensi_kurang = [
        bersihkan_level_item(x)
        for x in pecah_item(row.get("Kompetensi_Kurang", ""))
    ]
    
    blacklist_html = [
    "</div>",
    "<div>",
    "</span>",
    "<span>",
    "<br>",
    ]

    kompetensi_cocok = [
    x for x in _filter_item_valid(kompetensi_cocok)
    if x.strip().lower() not in blacklist_html
    and "<div" not in x.lower()
    and "</div" not in x.lower()
    ]

    kompetensi_kurang = [
    x for x in _filter_item_valid(kompetensi_kurang)
    if x.strip().lower() not in blacklist_html
    and "<div" not in x.lower()
    and "</div" not in x.lower()
    ]

    daftar_kompetensi: List[Tuple[str, str]] = []

    for item in kompetensi_cocok:
        daftar_kompetensi.append((item, "ok"))

    for item in kompetensi_kurang:
        daftar_kompetensi.append((item, "gap"))

    # Jika belum ada status cocok/gap, tampilkan daftar kompetensi netral.
    if not daftar_kompetensi:
        daftar_umum = _filter_item_valid(pecah_item(row.get("Daftar_Kompetensi", "")))
        for item in daftar_umum:
            daftar_kompetensi.append((item, "neutral"))

    return daftar_kompetensi


def _buat_html_kompetensi(daftar_kompetensi):

    items = []

    for item, kondisi in daftar_kompetensi:

        item_safe = nilai_aman(item)

        if kondisi == "ok":
            css_class = "competency-ok"
            label = '<span class="status-pill status-pill-ok">Sesuai</span>'

        elif kondisi == "gap":
            css_class = "competency-gap"
            label = '<span class="status-pill status-pill-gap">Belum terpenuhi</span>'

        else:
            css_class = "competency-neutral"
            label = ""

        items.append(
            f'<div class="competency-item {css_class}">{item_safe}{label}</div>'
        )

    if not items:
        return '<div class="competency-item competency-neutral">Belum ada data kompetensi.</div>'

    return "".join(items)


def _buat_html_rekomendasi(row) -> str:
    """Membuat HTML rekomendasi pelatihan."""
    rekomendasi = _filter_item_valid(pecah_item(row.get("Rekomendasi_Pelatihan", "")))

    # Hilangkan teks placeholder yang tidak perlu.
    rekomendasi = [
        x for x in rekomendasi
        if x.lower() not in ["belum ada rekomendasi pelatihan", "nan", "none", "null"]
    ]

    rekom_html = ""

    for i, item in enumerate(rekomendasi, start=1):
        rekom_html += f'<div class="rekom-item">{i}. {nilai_aman(item)}</div>'

    if not rekom_html:
        rekom_html = '<div class="rekom-item">Belum ada rekomendasi pelatihan.</div>'

    return rekom_html


def tampilkan_detail_pegawai(row):
    """Menampilkan detail pegawai yang dipilih dari tabel.

    Data yang ditampilkan:
    - Nama
    - NIP
    - Jabatan
    - Seksi / Unit_Es_IV
    - Unit III / Unit_Es_III
    - Kantor / Unit_Es_II
    - Skor kecocokan
    - Status kompetensi
    - Daftar kompetensi
    - Rekomendasi pelatihan
    """
    nama = nilai_aman(row.get("Nama_Pegawai", "-"))
    nip = nilai_aman(row.get("NIP_Panjang", "-"))
    jabatan = nilai_aman(row.get("Jabatan", "-"))
    seksi = nilai_aman(row.get("Unit_Es_IV", "-"))
    unit_iii = nilai_aman(row.get("Unit_Es_III", "-"))
    kantor = nilai_aman(row.get("Unit_Es_II", "-"))
    status = nilai_aman(row.get("Status_Kompetensi", "-"))
    skor = format_skor(row.get("Skor_Kecocokan_%", 0))

    jumlah_diuji = nilai_aman(row.get("Jumlah_Kompetensi_Diuji", "-"))
    jumlah_cocok = nilai_aman(row.get("Jumlah_Kompetensi_Cocok", "-"))

    daftar_kompetensi = _buat_daftar_kompetensi(row)
    item_html = _buat_html_kompetensi(daftar_kompetensi)
    rekom_html = _buat_html_rekomendasi(row)

    st.markdown(
        '<div class="section-title">👤 Detail Hasil Pencarian Pegawai</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="profile-card">
            <div class="profile-layout">
                <div class="profile-avatar">👤</div>
                <div>
                    <div class="profile-title">Data Pegawai</div>
                    <div class="profile-row"><span class="profile-label">Nama</span>: {nama}</div>
                    <div class="profile-row"><span class="profile-label">NIP</span>: {nip}</div>
                    <div class="profile-row"><span class="profile-label">Jabatan</span>: {jabatan}</div>
                    <div class="profile-row"><span class="profile-label">Seksi</span>: {seksi}</div>
                    <div class="profile-row"><span class="profile-label">Unit III</span>: {unit_iii}</div>
                    <div class="profile-row"><span class="profile-label">Kantor</span>: {kantor}</div>
                    <div class="small-note">
                        Skor kecocokan: {skor} | Status: {status}<br>
                        Kompetensi cocok: {jumlah_cocok} dari {jumlah_diuji}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_kompetensi, col_status = st.columns([1, 1])

    with col_kompetensi:
        st.markdown(
            f"""
            <div class="competency-box">
                <div class="competency-title">Kompetensi</div>
                {item_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_status:
        if html.unescape(status).strip().lower() == "kompeten":
            st.markdown(
                """
                <div class="status-good">
                    Pegawai sudah<br>kompeten
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="status-bad">
                    <div class="rekom-title">Rekomendasi Pelatihan</div>
                    {rekom_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# Komponen tambahan opsional
# ============================================================

def tampilkan_badge_status(status: str) -> str:
    """Menghasilkan badge HTML sederhana untuk status kompetensi."""
    status_safe = nilai_aman(status)
    status_plain = html.unescape(status_safe).strip().lower()

    if status_plain == "kompeten":
        return f'<span class="status-pill status-pill-ok">{status_safe}</span>'

    if status_plain == "tidak kompeten":
        return f'<span class="status-pill status-pill-gap">{status_safe}</span>'

    return f'<span class="status-pill">{status_safe}</span>'


def render_info_kecil(teks: str):
    """Menampilkan catatan kecil dengan style dashboard."""
    st.markdown(
        f'<div class="small-note">{html.escape(str(teks))}</div>',
        unsafe_allow_html=True,
    )
