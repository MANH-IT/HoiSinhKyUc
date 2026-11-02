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
if not exist "%TMP%" mkdir "%TMP%"
if not exist "%PIP_CACHE_DIR%" mkdir "%PIP_CACHE_DIR%"
if not exist "wheels" mkdir "wheels"

where py >nul 2>&1
set "HAS_PY=%errorlevel%"
where python >nul 2>&1
set "HAS_PYTHON=%errorlevel%"
if not "%HAS_PY%"=="0" if not "%HAS_PYTHON%"=="0" (
    echo Khong tim thay Python trong PATH.
    echo Hay cai Python 3.11 va chon "Add Python to PATH".
    pause
    exit /b 1
)

if not exist "%PY%" (
    echo [1/3] Tao moi truong ao .venv bang Python 3.11...
    if "%HAS_PY%"=="0" (
        py -3.11 -m venv "%VENV%" >nul 2>&1
        if errorlevel 1 py -3.10 -m venv "%VENV%" >nul 2>&1
    )
    if not exist "%PY%" python -m venv "%VENV%"
    if not exist "%PY%" goto :error
) else (
    echo [1/3] Da co moi truong ao .venv.
)

echo [2/3] Kiem tra thu vien...
"%PY%" -c "import cv2, numpy, streamlit" >nul 2>&1
if errorlevel 1 (
    echo Dang cai thu vien tu requirements.txt...
    "%PY%" -m pip install --retries 10 --timeout 60 --find-links wheels -r requirements.txt
    if errorlevel 1 goto :error
) else (
    echo Thu vien co ban da san sang.
)

if exist "models\GFPGANv1.4.pth" (
    if exist "models\RealESRGAN_x4plus.pth" (
        echo Da co model GFPGAN va Real-ESRGAN.
    ) else (
        echo Da co model GFPGAN, chua co Real-ESRGAN.
    )
) else (
    echo Chua co model AI. Chay: python scripts\download_models.py --all
)

if exist "models\colorization\colorization_release_v2.caffemodel" (
    echo Da co model to mau Caffe.
) else (
    echo Chua co model to mau. Chay: python scripts\download_models.py --colorization
)

"%PY%" -c "import basicsr" >nul 2>&1
if errorlevel 1 (
    if exist "models\GFPGANv1.4.pth" (
        echo Thu vien AI chua san sang. Chay setup_ai.bat de cai dat.
    )
)

echo [3/3] Khoi dong giao dien Streamlit...
"%PY%" -m streamlit run app\streamlit_app.py
goto :eof

:error
echo.
echo Co loi xay ra. Kiem tra Python, pip va ket noi mang.
pause
exit /b 1
