# Segmentation Sémantique d'Images Hyperspectrales 

## Description

Ce projet de fin d'études a été réalisé dans le cadre de la formation d'ingénieur en Big Data & Intelligence Artificielle à l'Université Internationale de Rabat (UIR). Il porte sur la segmentation sémantique d’images hyperspectrales à l’aide des **Graph Neural Networks** (GNN), avec un accent particulier sur la **sélection optimale des bandes spectrales** pour améliorer les performances et réduire la complexité.

---

## Objectifs

- Évaluer différentes méthodes de **sélection de bandes spectrales** : IoU, Coefficient de Déflexion.
- Optimiser la segmentation du jeu de données **Indian Pines** avec un minimum de bandes tout en maximisant la précision.
- Comparer les architectures **GCN**, **GAT** et **GraphSAGE** pour la segmentation hyperspectrale.
- Étudier la **robustesse** des architectures GNN en condition de données annotées limitées.
---

## Structure du Projet

```plaintext
├── 02_Donnees/                    # Chargement et traitement du dataset Indian Pines
├── 03_Selection_Bandes/           # Méthodes IoU One-vs-All, Worst-Case, Déflexion
├── 04_Modelisation_GCN/           # Scripts d'entraînement des modèles GCN
├── 05_Comparaison_Architectures/  # Comparaison GCN vs GAT vs GraphSAGE
├── utils/                         # Fonctions utilitaires partagées (graph construction, metrics...)
└── README.md                      # Présentation du projet

Méthodologie
Dataset utilisé : Indian Pines (145×145 pixels, 200 bandes, 16 classes)

Sélection de bandes :

IoU One-vs-All et IoU Worst-Case

Coefficient de Déflexion (DC)

Stratégies : Top-N et Equal Spacing

Conversion en graphe :

Adjacence spatiale ou hybride

Modèles testés :

GCN, GAT, GraphSAGE (1 à 3 couches)

Résultats
Meilleure architecture : GAT à 3 couches (20 bandes sélectionnées)

F1-score optimal : ≈ 0.7564

Méthode de sélection optimale : Coefficient de Déflexion avec Equal Spacing

Les GNN surpassent nettement les modèles MLP/SVM classiques, notamment avec peu de données annotées (5%)

Auteur
ESSAADI Hajar
Promotion 2025 - Université Internationale de Rabat
Filière : Big Data & Intelligence Artificielle
Encadrant : Pr. Mehdi Zakroum (TIC_Lab)