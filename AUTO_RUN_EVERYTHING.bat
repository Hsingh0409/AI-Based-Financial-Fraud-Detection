@echo off
REM ============================================================================
REM AUTOMATED COMPLETE SETUP AND RUN
REM Just double-click this file after installing Python
REM ============================================================================

cd /d "%~dp0"

echo.
echo ========================================
echo AUTOMATED SETUP AND RUN
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH"
    pause
    exit /b 1
)

echo [1/4] Python found
python --version
echo.

REM Install dependencies
echo [2/4] Installing dependencies...
python -m pip install --upgrade pip --quiet
python -m pip install pandas numpy scikit-learn torch matplotlib scipy kaggle jupyter --quiet
echo Dependencies installed
echo.

REM Verify Kaggle
echo [3/4] Verifying Kaggle API...
if exist "%USERPROFILE%\.kaggle\kaggle.json" (
    echo Kaggle API configured
) else (
    if exist "%HOME%/.kaggle/kaggle.json" (
        echo Kaggle API configured
    ) else (
        echo WARNING: Kaggle API not found but continuing...
    )
)
echo.

REM Launch Jupyter
echo [4/4] Launching Jupyter Notebook...
echo.
echo Browser will open automatically.
echo In Jupyter, click "Run All" to execute everything.
echo.
echo This window will stay open - Press Ctrl+C to stop Jupyter when done.
echo.

start "" jupyter notebook notebooks/demo.ipynb

pause
