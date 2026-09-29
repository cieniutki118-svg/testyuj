# SP5MIG Signal Hunter

Analizator sygnałów radiowych na Windows 11 (PySide6).

## Funkcje
- Auto-detekcja wejścia audio USB
- Duży waterfall + analizator widma FFT
- Dekoder CW (Morse) na żywo z auto-WPM
- Odszumianie spektralne (spectral subtraction, minimum statistics)
- Odsłuch RAW / DSP
- Nagrywanie WAV
- Wskaźnik poziomu + estymowany SNR
- Wybór wejścia i wyjścia audio
- Tryb DEMO: syntetyczny sygnał CW przez ten sam tor DSP

## Instalacja (Windows)
1. Zainstaluj Python 3.10+ (python.org, zaznacz "Add to PATH").
2. Uruchom `install.bat`.
3. Uruchom `start.bat`.

## Budowa EXE
Uruchom `build_exe.bat`. Wynik: `dist\SP5MIG_Signal_Hunter.exe`.

## Testy
venv\Scripts\activate
pytest

text
