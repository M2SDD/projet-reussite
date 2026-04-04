# Projet Réussite

Projet commun
- Algorithmique
- Programmation
- Mathématiques pour l'informatique

## Problématique
Est-il possible de prédire la réussite d'un apprenant en analysant ses traces numériques au sein de la plateforme ARCHE ?


## Précisions
Proposer un modèle qui permet de prédire la réussite ou l'échec d'un apprenant donné qui utilise l'espace ARCHE relativement à un cours donné.  
Plus précisément, comparer la technique de régression (multiple) avec une autre technique issue de l'Intelligence artificielle (IA) et proposer une solution motivée par la méthode d'évaluation que vous adoptez.  
Ne pas utiliser de notebook jupyter et privilégier la POO. Les scripts nécessaires seront placés dans src et appelés dans le `main.py`. Tkinter sera utilisé pour faire l'interface graphique de l'application.


## Etapes

### ETL
L'étude s'appuie sur des importations de données provenant de deux plateformes différentes :
1. Logs ARCHE
2. Gestion des notes

On dispose donc au départ de deux fichiers au format CSV, dans le dossier `data`, construits à partir des imports anonymisés.


**Logs**  
Le fichier contenant les logs représente les traces d'activités de chaque apprenant d'un cours sur la plateforme ARCHE. Les champs sont les suivants:
- heure : horodatage de l'activité
- pseudo : id de l'apprenant
- contexte: ressource objet de l'activité
- composant: type d'activité
- evenement: précisions sur l'activité


**Notes** (variable cible)  
Le fichier contenant les notes représente les notes obtenues par chaque apprenant.  
Les champs :
- pseudo : id de l'apprenant
- note : note obtenue


### Feature engineeiring
Il s'agit d'une part de calculer les indicateurs à étudier à partir des logs, et d'autre part d'effectuer une étude qui permet de retenir les meilleurs.


### Modélisation
Pour ce projet, l'utilisation de la régression linéaire multiple est imposée et vous comparerez avec une autre approche de Machine Learning au choix.


### Evaluation
Au terme de l'étude, il est important de mesurer et de comparer la qualité des modèle étudiés.


## Livrables
1. Un dossier technique (LaTeX) expliquant :
	- La conception du logiciel
	- La conception de l'interface utilisateur
	- Les principaux algorithmes
2. Un dossier d'analyse explicitant chacune des étapes du projet
3. Une distribution du code source permettant de tester l'application
4. Une soutenance orale (support en LaTeX)