@echo off
title Dashboard Evaluasi Kompetensi Pegawai v2.2

echo ============================================================
echo  Dashboard Evaluasi Kompetensi Pegawai v2.2
echo ============================================================
echo.

cd /d "C:\Users\d04nr\OneDrive - Kemenkeu\Kantor\Diklat\PJJ Data Analitik\Action Learning\Sharing\prj_v2.2\app_2.2.py"

echo Folder aktif:
cd
echo.

if not exist "app_2.2.py" (
    echo [ERROR] File app_2.2.py tidak ditemukan di folder ini.
    echo Pastikan semua file app_2.2 sudah diekstrak ke folder prj_v2.2.
    echo.
    pause
    exit /b
)

if exist "requirements_2.2.txt" (
    echo Mengecek / menginstall dependency dari requirements_2.2.txt...
    python -m pip install -r requirements_2.2.txt
    echo.
)

echo Menjalankan Streamlit...
echo.
python -m streamlit run app_2.2.py

echo.
pause
