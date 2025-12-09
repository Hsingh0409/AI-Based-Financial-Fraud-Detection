@echo off
REM This file navigates to the correct directory and launches Jupyter
REM You can double-click this file from Windows Explorer

cd /d "%~dp0"
echo Current directory: %CD%
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Please install Python first or run CLICK_ME_TO_SETUP.bat
    pause
    exit /b 1
)

echo Checking dependencies...
python -c "import jupyter" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies first...
    python -m pip install -r requirements.txt
)

echo.
echo Launching Jupyter Notebook...
echo The notebook will open in your browser.
echo.
echo Press Ctrl+C in this window to stop Jupyter when done.
echo.

jupyter notebook notebooks/demo.ipynb
pause
