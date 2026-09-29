@echo off
setlocal
cd /d %~dp0
echo === SP5MIG Signal Hunter - instalacja ===
where python >nul 2>nul
if errorlevel 1 (
    echo [BLAD] Python nie znaleziony w PATH.
    pause
    exit /b 1
)
if not exist venv (
    echo Tworze srodowisko wirtualne...
    python -m venv venv
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo [BLAD] Instalacja bibliotek nie powiodla sie.
    pause
    exit /b 1
)
echo.
echo Instalacja zakonczona pomyslnie.
pause
