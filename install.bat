@echo off
REM install.bat — Création de l'environnement virtuel et installation des dépendances

set VENV_DIR=.venv

echo ==^> Création de l'environnement virtuel dans '%VENV_DIR%'...
python -m venv %VENV_DIR%

echo ==^> Activation de l'environnement virtuel...
call %VENV_DIR%\Scripts\activate.bat

echo ==^> Mise à jour de pip...
pip install --upgrade pip

echo ==^> Installation des dépendances depuis requirements.txt...
pip install -r requirements.txt

echo.
echo Installation terminée.
echo Pour activer l'environnement manuellement : %VENV_DIR%\Scripts\activate.bat
