"""
Band Selection Methods - Projet de fin d'études
Méthodes de sélection de bandes spectrales pour images hyperspectrales

Ce module implémente différentes méthodes de sélection de bandes :
- IoU Global (One vs All)
- IoU par paires de classes (Worst Case)
- Coefficient de déflexion
"""

import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os


def calculer_chevauchement(classe_A_min, classe_A_max, classe_B_min, classe_B_max):
    """
    Calcule le chevauchement entre deux plages de valeurs.
    
    Args:
        classe_A_min, classe_A_max: Valeurs min et max pour la classe A
        classe_B_min, classe_B_max: Valeurs min et max pour la classe B
    
    Returns:
        Chevauchement normalisé (0 = aucun chevauchement, >0 = chevauchement)
    """
    # Calcul des bornes de chevauchement
    a = max(classe_A_min, classe_B_min)  # La plus grande des valeurs minimales
    b = min(classe_A_max, classe_B_max)  # La plus petite des valeurs maximales
    
    # Calcul du chevauchement brut
    c = b - a
    
    # Calcul de l'étendue totale
    etendue_totale = max(classe_A_max, classe_B_max) - min(classe_A_min, classe_B_min)
    
    # Normalisation du chevauchement
    if etendue_totale > 0:
        c_normalise = c / etendue_totale
    else:
        c_normalise = 0
    
    # Retourne max(0, c_normalise)
    return max(0, c_normalise)


def calculer_separabilite_paires_classes(pixels, classes, classes_uniques, class_names):
    """
    Calcule la séparabilité entre chaque paire de classes pour chaque bande spectrale.
    
    Args:
        pixels: Données spectrales (n_pixels x n_bandes)
        classes: Étiquettes de classe pour chaque pixel
        classes_uniques: Liste des classes uniques à considérer
        class_names: Noms des classes
    
    Returns:
        DataFrame contenant les résultats de séparabilité par paires
    """
    # Créer une liste pour stocker les résultats
    resultats_list = []
    
    # Nombre total d'itérations pour la barre de progression
    total_iterations = len(classes_uniques) * (len(classes_uniques) - 1) // 2 * pixels.shape[1]
    
    # Utiliser tqdm pour afficher une barre de progression
    with tqdm(total=total_iterations, desc="Calcul de séparabilité par paires") as pbar:
        # Pour chaque paire de classes
        for i, classe_A in enumerate(classes_uniques):
            for classe_B in classes_uniques[i+1:]:  # Ne considérer que les paires uniques
                # Obtenir les noms des classes
                nom_classe_A = class_names[classe_A] if classe_A < len(class_names) else f"Classe {classe_A}"
                nom_classe_B = class_names[classe_B] if classe_B < len(class_names) else f"Classe {classe_B}"
                
                # Créer les masques pour les deux classes
                mask_A = classes == classe_A
                mask_B = classes == classe_B
                
                # Pour chaque bande spectrale
                for bande in range(pixels.shape[1]):
                    # Extraire les valeurs de la bande pour les deux classes
                    valeurs_A = pixels[mask_A, bande]
                    valeurs_B = pixels[mask_B, bande]
                    
                    # Vérifier que les deux classes ont des pixels
                    if len(valeurs_A) > 0 and len(valeurs_B) > 0:
                        # Calculer les min et max pour chaque classe
                        classe_A_min = np.min(valeurs_A)
                        classe_A_max = np.max(valeurs_A)
                        classe_B_min = np.min(valeurs_B)
                        classe_B_max = np.max(valeurs_B)
                        
                        # Calculer le chevauchement normalisé
                        chevauchement = calculer_chevauchement(classe_A_min, classe_A_max, classe_B_min, classe_B_max)
                        
                        # Calculer la séparabilité (1 - chevauchement)
                        separabilite = 1 - chevauchement
                        
                        # Stocker les résultats dans la liste
                        resultats_list.append({
                            'ClasseA': int(classe_A),
                            'ClasseB': int(classe_B),
                            'NomClasseA': nom_classe_A,
                            'NomClasseB': nom_classe_B,
                            'Bande': int(bande),
                            'Separabilite': float(separabilite),
                            'Chevauchement': float(chevauchement)
                        })
                    
                    # Mettre à jour la barre de progression
                    pbar.update(1)
    
    # Créer un DataFrame à partir de la liste de résultats
    resultats_paires = pd.DataFrame(resultats_list)
    
    return resultats_paires


def analyser_worst_case(resultats_paires, top_n=20):
    """
    Analyse les bandes par leur worst-case de séparabilité et affiche les top N.
    
    Args:
        resultats_paires: DataFrame contenant les résultats de séparabilité
        top_n: Nombre de meilleures bandes à afficher
    
    Returns:
        DataFrame contenant les bandes triées par leur worst-case
    """
    print("Identification du worst-case pour chaque bande...")
    
    # Pour chaque bande, trouver le pire cas de séparabilité (minimum)
    worst_case_par_bande = (resultats_paires
                           .groupby('Bande')
                           .agg({
                               'Separabilite': 'min',  # Prendre la séparabilité minimum
                               'Chevauchement': 'max'   # Le chevauchement maximum correspondant
                           })
                           .reset_index())
    
    # Trier les bandes par leur worst-case de séparabilité (ordre décroissant)
    worst_case_par_bande = worst_case_par_bande.sort_values('Separabilite', ascending=False)
    
    # Afficher les top_n meilleures bandes selon leur worst-case
    print(f"\nTop {top_n} des bandes selon leur pire cas de séparabilité:")
    print(worst_case_par_bande.head(top_n))
    
    return worst_case_par_bande


def calculate_iou_global(image, labels, exclude_background=True):
    """
    Calcule l'IoU global (One vs All) pour chaque bande
    
    Parameters:
    -----------
    image : numpy.ndarray
        Image hyperspectrale (height, width, bands)
    labels : numpy.ndarray
        Étiquettes ground truth (height, width)
    exclude_background : bool
        Exclure la classe 0 (background)
        
    Returns:
    --------
    iou_scores : dict
        IoU score pour chaque bande
    best_bands : list
        Indices des meilleures bandes triées par IoU décroissant
    """
    # Vérification et ajustement de forme
    if image.shape[0] > image.shape[1] and image.shape[0] > image.shape[2]:
        image = np.transpose(image, (1, 2, 0))
    
    height, width, num_bands = image.shape
    
    # Aplatir les données
    image_flat = image.reshape(-1, num_bands)
    labels_flat = labels.reshape(-1)
    
    # Exclure le background si demandé
    if exclude_background:
        mask = labels_flat != 0
        image_flat = image_flat[mask]
        labels_flat = labels_flat[mask]
    
    # Classes uniques
    unique_classes = np.unique(labels_flat)
    if exclude_background and 0 in unique_classes:
        unique_classes = unique_classes[unique_classes != 0]
    
    iou_scores = {}
    
    print(f"Calcul IoU global pour {num_bands} bandes et {len(unique_classes)} classes...")
    
    for band_idx in tqdm(range(num_bands), desc="Calcul IoU par bande"):
        band_data = image_flat[:, band_idx]
        iou_sum = 0
        
        for class_id in unique_classes:
            # Créer les masques binaires
            y_true = (labels_flat == class_id).astype(int)
            
            # Segmentation simple par seuillage (médiane)
            threshold = np.median(band_data)
            y_pred = (band_data > threshold).astype(int)
            
            # Calculer IoU pour cette classe
            iou = jaccard_score(y_true, y_pred, average='binary', zero_division=0)
            iou_sum += iou
        
        # IoU moyen pour cette bande
        iou_scores[band_idx] = iou_sum / len(unique_classes)
    
    # Trier les bandes par IoU décroissant
    best_bands = sorted(iou_scores.keys(), key=lambda x: iou_scores[x], reverse=True)
    
    return iou_scores, best_bands


def calculate_deflection_coefficient(image, labels, exclude_background=True):
    """
    Calcule le coefficient de déflexion pour chaque bande
    
    Parameters:
    -----------
    image : numpy.ndarray
        Image hyperspectrale (height, width, bands)
    labels : numpy.ndarray
        Étiquettes ground truth (height, width)
    exclude_background : bool
        Exclure la classe 0 (background)
        
    Returns:
    --------
    deflection_scores : dict
        Score de déflexion pour chaque bande
    best_bands : list
        Indices des meilleures bandes triées par déflexion décroissante
    """
    # Vérification et ajustement de forme
    if image.shape[0] > image.shape[1] and image.shape[0] > image.shape[2]:
        image = np.transpose(image, (1, 2, 0))
    
    height, width, num_bands = image.shape
    
    # Aplatir les données
    image_flat = image.reshape(-1, num_bands)
    labels_flat = labels.reshape(-1)
    
    # Exclure le background si demandé
    if exclude_background:
        mask = labels_flat != 0
        image_flat = image_flat[mask]
        labels_flat = labels_flat[mask]
    
    # Classes uniques
    unique_classes = np.unique(labels_flat)
    if exclude_background and 0 in unique_classes:
        unique_classes = unique_classes[unique_classes != 0]
    
    deflection_scores = {}
    
    print(f"Calcul coefficient de déflexion pour {num_bands} bandes et {len(unique_classes)} classes...")
    
    for band_idx in tqdm(range(num_bands), desc="Calcul déflexion par bande"):
        band_data = image_flat[:, band_idx]
        
        # Calculer la moyenne globale
        mean_global = np.mean(band_data)
        
        # Calculer la variance intra-classe et inter-classe
        variance_intra = 0
        variance_inter = 0
        
        for class_id in unique_classes:
            class_mask = labels_flat == class_id
            class_data = band_data[class_mask]
            
            if len(class_data) > 0:
                mean_class = np.mean(class_data)
                var_class = np.var(class_data)
                n_class = len(class_data)
                
                # Variance intra-classe pondérée
                variance_intra += n_class * var_class
                
                # Variance inter-classe
                variance_inter += n_class * (mean_class - mean_global) ** 2
        
        # Normaliser par le nombre total d'échantillons
        variance_intra /= len(band_data)
        variance_inter /= len(band_data)
        
        # Coefficient de déflexion (ratio variance inter/intra)
        if variance_intra > 0:
            deflection_scores[band_idx] = variance_inter / variance_intra
        else:
            deflection_scores[band_idx] = 0
    
    # Trier les bandes par déflexion décroissante
    best_bands = sorted(deflection_scores.keys(), key=lambda x: deflection_scores[x], reverse=True)
    
    return deflection_scores, best_bands


def preparer_donnees_pour_analyse(image, labels, exclude_background=True):
    """
    Prépare les données pour l'analyse de sélection de bandes
    
    Returns:
    --------
    pixels : numpy.ndarray
        Données spectrales (n_pixels, n_bandes)
    classes : numpy.ndarray
        Étiquettes de classe
    classes_uniques : list
        Classes uniques
    class_names : list
        Noms des classes
    """
    # Vérification et ajustement de forme
    if image.shape[0] > image.shape[1] and image.shape[0] > image.shape[2]:
        image = np.transpose(image, (1, 2, 0))
    
    # Aplatir les données
    pixels = image.reshape(-1, image.shape[2])
    classes = labels.reshape(-1)
    
    # Exclure le background si demandé
    if exclude_background:
        mask = classes != 0
        pixels = pixels[mask]
        classes = classes[mask]
    
    # Classes uniques
    classes_uniques = np.unique(classes)
    if exclude_background and 0 in classes_uniques:
        classes_uniques = classes_uniques[classes_uniques != 0]
    
    # Noms des classes (Indian Pines)
    class_names = [
        "Background", "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
        "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed",
        "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean",
        "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"
    ]
    
    return pixels, classes, classes_uniques.tolist(), class_names


def select_bands_iou_pairwise(image, labels, top_n=20, exclude_background=True):
    """
    Sélection de bandes basée sur l'IoU par paires (méthode worst-case)
    
    Returns:
    --------
    selected_bands : list
        Indices des meilleures bandes
    results_df : DataFrame
        Résultats détaillés
    """
    # Préparer les données
    pixels, classes, classes_uniques, class_names = preparer_donnees_pour_analyse(
        image, labels, exclude_background
    )
    
    # Calculer la séparabilité par paires
    resultats_paires = calculer_separabilite_paires_classes(
        pixels, classes, classes_uniques, class_names
    )
    
    # Analyser le worst-case
    worst_case_df = analyser_worst_case(resultats_paires, top_n)
    
    # Sélectionner les meilleures bandes
    selected_bands = worst_case_df.head(top_n)['Bande'].tolist()
    
    return selected_bands, worst_case_df


def visualize_band_selection_results(iou_scores=None, deflection_scores=None, pairwise_results=None, top_n=20):
    """
    Visualise les résultats des différentes méthodes de sélection
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # IoU Global
    if iou_scores is not None:
        bands = list(iou_scores.keys())
        scores = list(iou_scores.values())
        top_bands = sorted(bands, key=lambda x: iou_scores[x], reverse=True)[:top_n]
        
        axes[0, 0].bar(range(len(scores)), scores, alpha=0.7)
        axes[0, 0].set_title('IoU Global par bande')
        axes[0, 0].set_xlabel('Indice de bande')
        axes[0, 0].set_ylabel('Score IoU')
        
        # Highlight top bands
        for band in top_bands:
            axes[0, 0].bar(band, iou_scores[band], color='red', alpha=0.8)
    
    # Coefficient de déflexion
    if deflection_scores is not None:
        bands = list(deflection_scores.keys())
        scores = list(deflection_scores.values())
        top_bands = sorted(bands, key=lambda x: deflection_scores[x], reverse=True)[:top_n]
        
        axes[0, 1].bar(range(len(scores)), scores, alpha=0.7)
        axes[0, 1].set_title('Coefficient de déflexion par bande')
        axes[0, 1].set_xlabel('Indice de bande')
        axes[0, 1].set_ylabel('Coefficient de déflexion')
        
        # Highlight top bands
        for band in top_bands:
            axes[0, 1].bar(band, deflection_scores[band], color='red', alpha=0.8)
    
    # IoU Pairwise (worst-case)
    if pairwise_results is not None:
        top_bands = pairwise_results.head(top_n)
        axes[1, 0].bar(range(len(top_bands)), top_bands['Separabilite'], alpha=0.7)
        axes[1, 0].set_title(f'Top {top_n} bandes (IoU Pairwise - Worst Case)')
        axes[1, 0].set_xlabel('Rang')
        axes[1, 0].set_ylabel('Séparabilité (worst-case)')
        
        # Add band indices as labels
        axes[1, 0].set_xticks(range(len(top_bands)))
        axes[1, 0].set_xticklabels(top_bands['Bande'].values, rotation=45)
    
    # Comparaison des méthodes
    if all(x is not None for x in [iou_scores, deflection_scores, pairwise_results]):
        iou_top = sorted(iou_scores.keys(), key=lambda x: iou_scores[x], reverse=True)[:top_n]
        defl_top = sorted(deflection_scores.keys(), key=lambda x: deflection_scores[x], reverse=True)[:top_n]
        pair_top = pairwise_results.head(top_n)['Bande'].tolist()
        
        methods = ['IoU Global', 'Déflexion', 'IoU Pairwise']
        band_sets = [set(iou_top), set(defl_top), set(pair_top)]
        
        # Intersection entre méthodes
        intersections = []
        for i, set1 in enumerate(band_sets):
            for j, set2 in enumerate(band_sets):
                if i <= j:
                    intersections.append(len(set1.intersection(set2)))
        
        axes[1, 1].text(0.1, 0.8, f"Bandes communes (top {top_n}):", transform=axes[1, 1].transAxes, fontsize=12, weight='bold')
        axes[1, 1].text(0.1, 0.7, f"IoU Global ∩ Déflexion: {len(band_sets[0].intersection(band_sets[1]))}", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f"IoU Global ∩ IoU Pairwise: {len(band_sets[0].intersection(band_sets[2]))}", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f"Déflexion ∩ IoU Pairwise: {len(band_sets[1].intersection(band_sets[2]))}", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, f"Toutes méthodes: {len(band_sets[0].intersection(band_sets[1]).intersection(band_sets[2]))}", transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Comparaison des méthodes')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# PARTIE TEST/DÉMONSTRATION
# ============================================================================

def demo_band_selection(image, labels, save_results=True):
    """
    Démonstration complète des méthodes de sélection de bandes
    """
    print("=== Démonstration des méthodes de sélection de bandes ===\n")
    
    # 1. IoU Global
    print("1. Calcul IoU Global (One vs All)...")
    iou_scores, iou_best = calculate_iou_global(image, labels)
    print(f"Top 5 bandes IoU Global: {iou_best[:5]}")
    
    # 2. Coefficient de déflexion
    print("\n2. Calcul coefficient de déflexion...")
    deflection_scores, deflection_best = calculate_deflection_coefficient(image, labels)
    print(f"Top 5 bandes déflexion: {deflection_best[:5]}")
    
    # 3. IoU Pairwise
    print("\n3. Calcul IoU Pairwise (Worst Case)...")
    pairwise_bands, pairwise_results = select_bands_iou_pairwise(image, labels, top_n=20)
    print(f"Top 5 bandes IoU Pairwise: {pairwise_bands[:5]}")
    
    # 4. Visualisation
    print("\n4. Génération des visualisations...")
    visualize_band_selection_results(iou_scores, deflection_scores, pairwise_results)
    
    # 5. Sauvegarde des résultats
    if save_results:
        results = {
            'iou_global_top20': iou_best[:20],
            'deflection_top20': deflection_best[:20],
            'pairwise_top20': pairwise_bands[:20]
        }
        
        np.save('band_selection_results.npy', results)
        print("\nRésultats sauvegardés dans 'band_selection_results.npy'")
    
    return {
        'iou_global': (iou_scores, iou_best),
        'deflection': (deflection_scores, deflection_best),
        'pairwise': (pairwise_bands, pairwise_results)
    }