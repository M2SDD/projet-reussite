#!/usr/bin/env bash
# run.sh — Lancement de l'application via l'environnement virtuel

set -e

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Environnement virtuel introuvable. Lancez d'abord ./install.sh"
    exit 1
fi

echo "==> Activation de l'environnement virtuel..."
source "$VENV_DIR/bin/activate"

echo "==> Lancement de main.py..."
python main.py
