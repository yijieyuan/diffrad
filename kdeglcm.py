import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

def compute_glcm_kde(images, binNum=64, delta_r=1, delta_c=1):
    """
    Compute Gray Level Co-occurrence Matrix (GLCM) using Kernel Density Estimation (KDE).
    
    Parameters:
    -----------
    images : numpy.ndarray or torch.Tensor
        Input images with shape (batch_size, height, width)
        Values should be normalized between 0 and 1
    binNum : int, optional
        Number of bins for the GLCM (default: 64)
    delta_r : int, optional
        Row offset for co-occurrence (default: 1)
    delta_c : int, optional
        Column offset for co-occurrence (default: 1)
        
    Returns:
    --------
    glcm : numpy.ndarray
        Gray Level Co-occurrence Matrix with shape (binNum, binNum)
    """
    # Get device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensor if input is numpy array
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).float().to(device)
    else:
        images = images.float().to(device)
    
    # Ensure values are between 0 and 1
    if images.max() > 1.0:
        images = images / 255.0
    
    # Parameters for KDE
    binWidth = 1 / binNum
    sigma = 0.5 * binWidth
    
    # Get dimensions
    batch_size, height, width = images.shape
    
    # Initialize tensor to store KDE values
    kde_tensor = torch.zeros(batch_size, height, width, binNum).to(device)
    
    # Predefine bin matrix
    predefine_bin_matrix = torch.linspace(binWidth, 1 - binWidth, steps=binNum).view(1, 1, 1, binNum).to(device)
    predefine_bin_matrix = predefine_bin_matrix.repeat(batch_size, height, width, 1)
    
    # Calculate distance tensor
    distance_tensor = predefine_bin_matrix - images.view(batch_size, height, width, 1).repeat(1, 1, 1, binNum)
    
    # Calculate KDE values
    kde_values = torch.exp(-(distance_tensor)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    kde_tensor = kde_values / kde_values.sum(dim=3, keepdim=True)
    
    # Extract slices from kde_tensor for co-occurrence
    kde1 = kde_tensor[:, delta_r:, delta_c:, :]
    kde2 = kde_tensor[:, :-delta_r, :-delta_c, :]
    
    # Compute GLCM using einsum for efficient computation
    glcm_kde = torch.sum(torch.einsum('ijkl,ijkm->ijklm', kde1, kde2), dim=(0, 1)).cpu().numpy()
    
    # Normalize by the number of pixel pairs
    glcm_kde = glcm_kde / ((height - delta_r) * (width - delta_c) * batch_size)
    
    return glcm_kde