import math
import pandas as pd
import pytest

from modul_tampilan_22 import (
    nilai_aman,
    angka_aman,
    format_skor,
    _filter_item_valid,
    _buat_daftar_kompetensi,
    _buat_html_kompetensi,
    _buat_html_rekomendasi,
)


def test_nilai_aman():
    # Menguji nilai string valid
    assert nilai_aman("Testing") == "Testing"
    
    # Menguji karakter HTML (escaping)
    assert nilai_aman("<script>alert('x')</script>") == "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;"
    
    # Menguji penanganan null/kosong
    assert nilai_aman(None) == "-"
    assert nilai_aman(math.nan) == "-"
    assert nilai_aman(pd.NA) == "-"
    assert nilai_aman("") == "-"
    assert nilai_aman("   ") == "-"
    
    # Menguji penanganan teks "NaN"
    assert nilai_aman("NaN") == "-"
    assert nilai_aman("null") == "-"


def test_angka_aman():
    assert angka_aman("100.5") == 100.5
    assert angka_aman(85) == 85.0
    
    # Nilai invalid harus jatuh ke nilai default
    assert angka_aman("invalid_number") == 0.0
    assert angka_aman(math.nan) == 0.0
    assert angka_aman(None) == 0.0
    
    # Default kustom
    assert angka_aman(None, default=50.0) == 50.0


def test_format_skor():
    # Harus membuang desimal jika angka bulat
    assert format_skor(100.0) == "100%"
    assert format_skor(85) == "85%"
    
    # Harus menampilkan 2 desimal jika bukan angka bulat
    assert format_skor(85.555) == "85.55%"
    
    # Penanganan invalid
    assert format_skor("N/A") == "0%"


def test_filter_item_valid():
    raw_items = [
        "Analisis Data",
        "-",
        "NaN",
        "  ",
        "<div>",
        "</div>",
        "Manajemen Proyek"
    ]
    
    filtered = _filter_item_valid(raw_items)
    
    assert len(filtered) == 2
    assert "Analisis Data" in filtered
    assert "Manajemen Proyek" in filtered


def test_buat_daftar_kompetensi():
    row_data = {
        "Kompetensi_Cocok": "Analisis Data lv 2 | Presentasi",
        "Kompetensi_Kurang": "Manajemen Proyek (butuh level 3, saat ini 1)",
        "Daftar_Kompetensi": "Analisis Data | Presentasi | Manajemen Proyek",
    }
    
    daftar = _buat_daftar_kompetensi(row_data)
    
    assert len(daftar) == 3
    assert daftar[0] == ("Analisis Data lv 2", "ok")
    assert daftar[1] == ("Presentasi", "ok")
    assert daftar[2] == ("Manajemen Proyek lv 3", "gap")
    
    # Kasus saat tidak ada data Cocok / Kurang
    row_empty = {
        "Daftar_Kompetensi": "Komunikasi | Kepemimpinan"
    }
    daftar_netral = _buat_daftar_kompetensi(row_empty)
    assert daftar_netral == [("Komunikasi", "neutral"), ("Kepemimpinan", "neutral")]


def test_buat_html_kompetensi():
    daftar = [("Skill A", "ok"), ("Skill B", "gap"), ("Skill C", "neutral")]
    html_output = _buat_html_kompetensi(daftar)
    
    assert "competency-ok" in html_output
    assert "competency-gap" in html_output
    assert "competency-neutral" in html_output


def test_buat_html_rekomendasi():
    row_data = {"Rekomendasi_Pelatihan": "Pelatihan Data Analytics | Pelatihan Leadership"}
    html_output = _buat_html_rekomendasi(row_data)
    
    assert "1. Pelatihan Data Analytics" in html_output
    assert "2. Pelatihan Leadership" in html_output