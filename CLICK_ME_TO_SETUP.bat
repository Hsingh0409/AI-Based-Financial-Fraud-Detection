@echo off
REM This file navigates to the correct directory and runs setup
REM You can double-click this file from Windows Explorer

cd /d "%~dp0"
echo Current directory: %CD%
echo.
echo Running setup...
echo.
call setup.bat
