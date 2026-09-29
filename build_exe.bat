@echo off
setlocal
cd /d %~dp0
if not exist venv\Scripts\activate.bat (
    echo [BLAD] Brak venv. Uruchom najpierw install.bat
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
pip install pyinstaller
pyinstaller --noconfirm --clean --onefile --windowed ^
    --name SP5MIG_Signal_Hunter ^
    --hidden-import scipy.signal ^
    --hidden-import scipy.fft ^
    --hidden-import scipy.special ^
    --hidden-import scipy._lib.messagestream ^
    --collect-submodules sounddevice ^
    --collect-submodules soundfile ^
    main.py
if errorlevel 1 (
    echo [BLAD] PyInstaller nie powiodl sie.
    pause
    exit /b 1
)
echo.
echo EXE: dist\SP5MIG_Signal_Hunter.exe
pause
