# Projet Réussite

Système de prédiction et d'analyse de la réussite étudiante à partir des exports Arche anonymisés.

Projet commun aux matières :
- Algorithmique
- Programmation avancée (Python)
- Mathématiques pour l'informatique

---

## Prérequis

- Python 3.9+ (développé sous 3.12.10)
- `pip` disponible dans le PATH

---

## Installation

### Linux / macOS

```bash
chmod +x install.sh
./install.sh
```

### Windows

```bat
install.bat
```

Ces scripts créent un environnement virtuel `.venv` à la racine du projet, mettent pip à jour et installent toutes les dépendances listées dans `requirements.txt`.

---

## Lancement

### Linux / macOS

```bash
chmod +x run.sh
./run.sh
```

### Windows

```bat
run.bat
```

L'interface graphique (Tkinter) se lance automatiquement.

---

## Structure du projet

```
projet_reussite/
├── data/               Données sources (exports Arche)
├── output/             Résultats générés (graphiques, CSV)
├── pictures/           Illustrations statiques
├── rapport/            Dossiers technique et analyse (LaTeX)
├── src/                Code source Python
├── tests/              Tests unitaires
├── main.py             Point d'entrée de l'application
├── requirements.txt    Dépendances Python
├── install.sh / .bat   Scripts d'installation
└── run.sh / .bat       Scripts de lancement
```

---

## Auteur

Matthieu PELINGRE — M2 SDD, 2024-2025
