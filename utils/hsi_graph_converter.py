"""
Utilitaire pour convertir des images hyperspectrales en graphes
Utilisé dans le projet de segmentation sémantique d'images hyperspectrales
"""

import numpy as np
import torch
import scipy.io as sio
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import os

def load_indian_pines(dataset_path):
    """
    Charge le dataset Indian Pines
    
    Args:
        dataset_path: Chemin vers les fichiers .mat
        
    Returns:
        image: Image hyperspectrale (height, width, bands)
        gt: Ground truth labels (height, width)
    """
    # Chargement de l'image hyperspectrale corrigée
    data = sio.loadmat(os.path.join(dataset_path, 'Indian_pines_corrected.mat'))
    image = data['indian_pines_corrected']
    
    # Chargement des étiquettes (ground truth)
    gt_data = sio.loadmat(os.path.join(dataset_path, 'Indian_pines_gt.mat'))
    gt = gt_data['indian_pines_gt']
    
    return image, gt

def create_hyperspectral_graph(image, labels, selected_bands=None, dilation=1, connectivity_type='8-connectivity'):
    """
    Crée un graphe à partir d'une image hyperspectrale.
    
    Args:
        image: Image hyperspectrale de forme (height, width, bands)
        labels: Étiquettes de classe pour chaque pixel
        selected_bands: Liste des indices des bandes à utiliser (si None, toutes les bandes)
        dilation: Rayon de voisinage pour la connectivité
        connectivity_type: Type de connectivité ('4-connectivity' ou '8-connectivity')
        
    Returns:
        data: Objet Data de PyTorch Geometric
    """
    # Vérification et ajustement de la forme de l'image
    if image.shape[0] > image.shape[1] and image.shape[0] > image.shape[2]:
        # Si c'est (bands, height, width) -> transpose à (height, width, bands)
        image = np.transpose(image, (1, 2, 0))
    
    height, width, num_bands = image.shape
    num_pixels = height * width
    
    # Sélection des bandes spectrales
    if selected_bands is not None:
        features = image[:, :, selected_bands]
    else:
        features = image
    
    # Reshape et normalisation
    X = features.reshape(num_pixels, -1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    x = torch.FloatTensor(X)
    
    # Création des arêtes selon le type de connectivité
    edge_list = []
    
    if connectivity_type == '4-connectivity':
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    else:  # 8-connectivity
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), 
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    # Créer les arêtes avec dilation
    for i in range(height):
        for j in range(width):
            pixel_idx = i * width + j
            for di, dj in directions:
                for d in range(1, dilation + 1):
                    ni, nj = i + di * d, j + dj * d
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor_idx = ni * width + nj
                        edge_list.append((pixel_idx, neighbor_idx))
    
    # Conversion en tensors PyTorch
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    y = torch.tensor(labels.reshape(-1), dtype=torch.long)
    
    # Création de l'objet Data
    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = num_pixels
    data.height = height
    data.width = width
    
    return data

def save_graph(data, output_path, filename):
    """
    Sauvegarde un graphe
    
    Args:
        data: Objet Data de PyTorch Geometric
        output_path: Dossier de destination
        filename: Nom du fichier (sans extension)
    """
    full_path = os.path.join(output_path, f"{filename}.pt")
    torch.save(data, full_path)
    print(f"Graphe sauvegardé: {full_path}")

def create_graph_with_selected_bands(image, labels, band_indices, config_name):
    """
    Fonction simplifiée pour créer un graphe avec des bandes sélectionnées
    
    Args:
        image: Image hyperspectrale
        labels: Ground truth
        band_indices: Liste des indices de bandes à utiliser
        config_name: Nom de la configuration (ex: "iou_global", "deflection")
        
    Returns:
        data: Objet graphe PyTorch Geometric
    """
    return create_hyperspectral_graph(
        image=image,
        labels=labels,
        selected_bands=band_indices,
        dilation=1,
        connectivity_type='8-connectivity'
    )