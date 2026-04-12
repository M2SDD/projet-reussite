#!/usr/bin/env bash
# install.sh — Création de l'environnement virtuel et installation des dépendances

set -e

VENV_DIR=".venv"

echo "==> Création de l'environnement virtuel dans '$VENV_DIR'..."
python3 -m venv "$VENV_DIR"

echo "==> Activation de l'environnement virtuel..."
source "$VENV_DIR/bin/activate"

echo "==> Mise à jour de pip..."
pip install --upgrade pip

echo "==> Installation des dépendances depuis requirements.txt..."
pip install -r requirements.txt

echo ""
echo "Installation terminée."
echo "Pour activer l'environnement manuellement : source $VENV_DIR/bin/activate"
