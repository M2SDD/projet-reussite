#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de vérification de la structure du projet
"""

import sys
import os

def verify_structure():
    """Vérifie la structure complète du projet"""

    print("=== VERIFICATION DE LA STRUCTURE DU PROJET ===\n")

    # 1. Vérifier que src/ existe avec __init__.py
    print("1. Vérification du répertoire src/")
    if os.path.isdir("src"):
        print("   ✓ Répertoire src/ existe")
    else:
        print("   ✗ Répertoire src/ manquant")
        return False

    if os.path.isfile("src/__init__.py"):
        print("   ✓ src/__init__.py existe")
    else:
        print("   ✗ src/__init__.py manquant")
        return False

    # 2. Vérifier que tous les modules existent
    print("\n2. Vérification des fichiers de modules")
    modules = ["config.py", "data_loader.py", "data_processor.py", "visualizer.py"]
    for module in modules:
        if os.path.isfile(f"src/{module}"):
            print(f"   ✓ src/{module} existe")
        else:
            print(f"   ✗ src/{module} manquant")
            return False

    # 3. Vérifier que tous les modules sont importables
    print("\n3. Vérification des imports de modules")
    try:
        import src
        print("   ✓ Package src importable")
    except ImportError as e:
        print(f"   ✗ Erreur d'import du package src: {e}")
        return False

    try:
        import src.config
        print("   ✓ src.config importable")
    except ImportError as e:
        print(f"   ✗ Erreur d'import de src.config: {e}")
        return False

    try:
        import src.data_loader
        print("   ✓ src.data_loader importable")
    except ImportError as e:
        print(f"   ✗ Erreur d'import de src.data_loader: {e}")
        return False

    try:
        import src.data_processor
        print("   ✓ src.data_processor importable")
    except ImportError as e:
        print(f"   ✗ Erreur d'import de src.data_processor: {e}")
        return False

    try:
        import src.visualizer
        print("   ✓ src.visualizer importable")
    except ImportError as e:
        print(f"   ✗ Erreur d'import de src.visualizer: {e}")
        return False

    # 4. Vérifier que les classes peuvent être importées depuis le package
    print("\n4. Vérification des imports de classes depuis le package")
    try:
        from src import Config, DataLoader, DataProcessor, Visualizer
        print("   ✓ Toutes les classes importables depuis src")
    except ImportError as e:
        print(f"   ✗ Erreur d'import des classes: {e}")
        return False

    # 5. Vérifier que requirements.txt existe
    print("\n5. Vérification de requirements.txt")
    if os.path.isfile("requirements.txt"):
        print("   ✓ requirements.txt existe")
        with open("requirements.txt", "r") as f:
            content = f.read()
            required_deps = ["pandas", "numpy", "scikit-learn", "matplotlib", "seaborn"]
            for dep in required_deps:
                if dep in content:
                    print(f"   ✓ {dep} présent dans requirements.txt")
                else:
                    print(f"   ✗ {dep} manquant dans requirements.txt")
                    return False
    else:
        print("   ✗ requirements.txt manquant")
        return False

    # 6. Vérifier que les classes OOP sont définies (pas de fonctions isolées)
    print("\n6. Vérification de la structure OOP")
    try:
        from src import Config, DataLoader, DataProcessor, Visualizer

        # Vérifier que ce sont bien des classes
        if isinstance(Config, type):
            print("   ✓ Config est une classe")
        else:
            print("   ✗ Config n'est pas une classe")
            return False

        if isinstance(DataLoader, type):
            print("   ✓ DataLoader est une classe")
        else:
            print("   ✗ DataLoader n'est pas une classe")
            return False

        if isinstance(DataProcessor, type):
            print("   ✓ DataProcessor est une classe")
        else:
            print("   ✗ DataProcessor n'est pas une classe")
            return False

        if isinstance(Visualizer, type):
            print("   ✓ Visualizer est une classe")
        else:
            print("   ✗ Visualizer n'est pas une classe")
            return False

        # Vérifier que les classes peuvent être instanciées
        config = Config()
        loader = DataLoader()
        processor = DataProcessor()
        visualizer = Visualizer()
        print("   ✓ Toutes les classes peuvent être instanciées")

    except Exception as e:
        print(f"   ✗ Erreur lors de la vérification OOP: {e}")
        return False

    print("\n=== TOUTES LES VERIFICATIONS ONT REUSSI ===")
    return True

if __name__ == "__main__":
    success = verify_structure()
    sys.exit(0 if success else 1)
