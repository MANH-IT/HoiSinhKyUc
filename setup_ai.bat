@echo off
setlocal
cd /d "%~dp0"

set "VENV=.venv"
set "PY=%VENV%\Scripts\python.exe"
set "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "TMP=%~dp0.tmp"
set "TEMP=%~dp0.tmp"
set "PIP_CACHE_DIR=%~dp0.pip-cache"
set "PIP_ARGS=--retries 10 --timeout 60 --find-links wheels"
if not exist "%TMP%" mkdir "%TMP%"
if not exist "%PIP_CACHE_DIR%" mkdir "%PIP_CACHE_DIR%"
if not exist "wheels" mkdir "wheels"

where py >nul 2>&1
if errorlevel 1 (
    echo Khong tim thay py launcher cua Python.
    echo Hay cai Python 3.11 va bat "py launcher".
    pause
    exit /b 1
)

set "VENV_OK=0"
if exist "%PY%" (
    "%PY%" -c "import sys; sys.exit(0 if sys.version_info[:2] in {(3, 10), (3, 11)} else 1)" >nul 2>&1
    if not errorlevel 1 set "VENV_OK=1"
)

if "%VENV_OK%"=="1" goto :venv_ready

if exist "%VENV%" (
    echo Moi truong .venv hien tai khong dung Python 3.10/3.11.
    echo basicsr 1.4.2 khong cai duoc tren Python 3.13.
    choice /C YN /M "Tao lai .venv bang Python 3.11"
    if errorlevel 2 exit /b 1
    rmdir /s /q "%VENV%"
)

echo [1/5] Tao .venv bang Python 3.11...
py -3.11 -m venv "%VENV%"
if errorlevel 1 py -3.10 -m venv "%VENV%"
if errorlevel 1 (
    echo Khong tao duoc venv bang Python 3.11 hoac 3.10.
    echo Hay cai Python 3.11 hoac 3.10.
    pause
    exit /b 1
)

:venv_ready
echo [2/5] Cap nhat pip, setuptools, wheel...
"%PY%" -m pip install %PIP_ARGS% --upgrade pip setuptools wheel
if errorlevel 1 goto :error

echo [3/5] Cai thu vien co ban...
"%PY%" -m pip install %PIP_ARGS% -r requirements.txt
if errorlevel 1 goto :error

echo [4/5] Cai thu vien AI...
"%PY%" -m pip install %PIP_ARGS% -r requirements-ai.txt
if errorlevel 1 goto :error

echo [5/5] Va tuong thich basicsr / torchvision...
"%PY%" scripts\patch_basicsr.py
if errorlevel 1 goto :error

"%PY%" -c "import cv2, torch, torchvision, gfpgan, basicsr, realesrgan; print('Kiem tra AI: OK')"
if errorlevel 1 goto :error

echo.
echo Cai dat AI hoan tat.
echo Chay run.bat de mo giao dien.
pause
exit /b 0

:error
echo.
echo Cai dat that bai. Xem loi phia tren.
pause
exit /b 1
