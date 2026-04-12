@echo off
REM run.bat — Lancement de l'application via l'environnement virtuel

set VENV_DIR=.venv

if not exist %VENV_DIR%\Scripts\activate.bat (
    echo Environnement virtuel introuvable. Lancez d'abord install.bat
    exit /b 1
)

echo ==^> Activation de l'environnement virtuel...
call %VENV_DIR%\Scripts\activate.bat

echo ==^> Lancement de main.py...
python main.py
