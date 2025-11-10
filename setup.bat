@echo off
REM Setup script for Binary Algos Project
REM Requires Python 3.11.x and pip

echo ========================================
echo Binary Algos - Setup Script
echo ========================================
echo.

REM Check if Python is installed and find Python 3.11
echo [1/4] Checking Python...

set PYTHON_CMD=
set PYTHON_VERSION=

REM Try py launcher with -3.11 flag first (most reliable)
py -3.11 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py -3.11
    for /f "tokens=2" %%i in ('py -3.11 --version 2^>^&1') do set PYTHON_VERSION=%%i
    goto found_python
)

REM Try python3.11 command
python3.11 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python3.11
    for /f "tokens=2" %%i in ('python3.11 --version 2^>^&1') do set PYTHON_VERSION=%%i
    goto found_python
)

REM Try default python and check if it's 3.11
python --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    for /f "tokens=1,2 delims=." %%a in ("!PYTHON_VERSION!") do (
        if "%%a.%%b"=="3.11" (
            set PYTHON_CMD=python
            goto found_python
        )
    )
)

REM Python 3.11 not found
echo [ERROR] Python 3.11 not found!
echo.
echo You have multiple Python versions. Please install Python 3.11.x from:
echo https://www.python.org/downloads/
echo.
echo Make sure to check "Add Python to PATH" during installation.
echo.
echo Or use Python Launcher: py -3.11
pause
exit /b 1

:found_python
echo Current Python: %PYTHON_CMD%
echo Python version: %PYTHON_VERSION%

REM Verify it's 3.11.x
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

if not "%MAJOR%.%MINOR%"=="3.11" (
    echo [ERROR] Python 3.11.x required but found Python %PYTHON_VERSION%
    pause
    exit /b 1
)

echo [OK] Python 3.11 is ready!

REM Check if pip is available
echo.
echo [2/4] Checking pip...
%PYTHON_CMD% -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] pip not found!
    echo Please install pip: %PYTHON_CMD% -m ensurepip --upgrade
    pause
    exit /b 1
)
echo [OK] pip is ready!

REM Create virtual environment
echo.
echo [3/4] Creating virtual environment...
if exist venv (
    echo Existing venv detected. Removing...
    rmdir /s /q venv
)

echo Creating venv with %PYTHON_CMD%... (should take 5-10 seconds)
%PYTHON_CMD% -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment!
    pause
    exit /b 1
)
echo [OK] Virtual environment created! (venv is small, around 10-20 MB)

:install_deps
REM Activate venv and install dependencies
echo.
echo [4/4] Installing dependencies from requirements.txt...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)

echo Upgrading pip...
python -m pip install --upgrade pip
echo.
echo ========================================
echo Installing packages from requirements.txt
echo This will take 5-15 minutes (downloading ~2-3 GB)
echo Large packages: JAX, TensorFlow, PyTorch
echo ========================================
echo.
echo Progress will show each package being downloaded and installed...
echo.
pip install -v -r requirements.txt

if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Python version: %PYTHON_VERSION%
echo Virtual environment: venv
echo.
echo Usage:
echo   1. Activate venv: venv\Scripts\activate
echo   2. Run Python code: python your_script.py
echo   3. Deactivate venv: deactivate
echo.
echo Installed packages:
pip list
echo.
pause