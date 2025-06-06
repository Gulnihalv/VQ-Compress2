"""
Görüntü kalitesi değerlendirme metrikleri
"""

import torch
import numpy as np
import math
from torch.nn import functional as F


def psnr(original, reconstructed):
    """
    İki görüntü arasındaki Peak Signal-to-Noise Ratio (PSNR) hesaplar
    
    Args:
        original (torch.Tensor): Orijinal görüntü, [0,1] aralığında
        reconstructed (torch.Tensor): Yeniden oluşturulan görüntü, [0,1] aralığında
        
    Returns:
        float: PSNR değeri (dB)
    """
    # Tensor'ları CPU'ya taşı ve numpy'a dönüştür
    if torch.is_tensor(original):
        original = original.detach().cpu().numpy()
    if torch.is_tensor(reconstructed):
        reconstructed = reconstructed.detach().cpu().numpy()
    
    # Görüntü tensör değerlerinin [0, 1] aralığında olduğunu varsayıyoruz
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 1.0
    psnr_value = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr_value


# def _gaussian_kernel(size=11, sigma=1.5):
#     """SSIM hesaplamak için Gaussian çekirdeği oluşturur"""
#     coords = torch.arange(size, dtype=torch.float)
#     coords -= size // 2
    
#     g = coords ** 2
#     g = torch.exp(-(g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma ** 2))
    
#     g /= g.sum()
#     return g.unsqueeze(0).unsqueeze(0)


def ms_ssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    """
    Multi-scale Structural Similarity Index Measure
    
    Args:
        img1 (torch.Tensor): Images with shape [N, C, H, W], range [0, 1]
        img2 (torch.Tensor): Images with shape [N, C, H, W], range [0, 1]
        window_size (int): Window size for SSIM calculation
        size_average (bool): Size averaging flag
        val_range (float): Value range of input images (usually 1.0 or 255)
        normalize (bool): Whether to normalize the SSIM by weights sum
        
    Returns:
        torch.Tensor: MS-SSIM value
    """
    import torch
    import torch.nn.functional as F
    from math import exp
    
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size(0)
    
    min_size = min(img1.size(-2), img1.size(-1))
    if min_size < (2 ** (levels - 1) + 1):
        # If image is too small for full multi-scale calculation
        # Resize weights to match available levels
        max_level = max(1, int(torch.log2(torch.tensor(min_size, dtype=torch.float32)).item()))
        weights = weights[:max_level]
        # Normalize the weights
        weights = weights / weights.sum()
        levels = weights.size(0)
    
    # Print tensor shapes and weights for debugging
    # print(f"img1 shape: {img1.shape}, img2 shape: {img2.shape}")
    # print(f"weights: {weights}, levels: {levels}")
    
    if val_range is None:
        val_range = 1.0  # Default value range for normalized images
    
    mssim = []
    mcs = []
    
    # Generate Gaussian kernel
    sigma = 1.5
    gauss_kernel_size = window_size
    
    # Create a 1D Gaussian kernel
    _1D_window = torch.Tensor([exp(-(x - gauss_kernel_size//2)**2/(2*sigma**2)) 
                            for x in range(gauss_kernel_size)]).to(device)
    _1D_window = _1D_window / _1D_window.sum()
    
    # Create 2D Gaussian kernel
    _2D_window = _1D_window.unsqueeze(1) @ _1D_window.unsqueeze(0)
    window = _2D_window.expand(img1.size(1), 1, gauss_kernel_size, gauss_kernel_size).contiguous()
    
    for i in range(levels):
        # Calculate SSIM and contrast sensitivity (CS) for this scale
        ssim_val, cs = _ssim(img1, img2, window, gauss_kernel_size, val_range, size_average)
        mssim.append(ssim_val)
        mcs.append(cs)
        
        # Downsample for next scale
        if i < levels - 1:
            padding = (img1.shape[2] % 2, img1.shape[3] % 2)
            img1 = F.avg_pool2d(img1, kernel_size=2, padding=padding)
            img2 = F.avg_pool2d(img2, kernel_size=2, padding=padding)
    
    # Convert lists to tensors
    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)
    
    # Calculate weighted MS-SSIM
    # Use einsum for safer multiplication
    pow1 = torch.pow(mcs[:levels-1], weights[:levels-1].unsqueeze(1))
    pow2 = torch.pow(mssim[levels-1], weights[levels-1])
    
    # Use torch.prod on the dimension with weights
    output = torch.prod(pow1, dim=0) * pow2
    
    # Average if needed
    if size_average:
        output = output.mean()
        
    return output


def _ssim(img1, img2, window, window_size, val_range, size_average):
    """
    Calculate SSIM index for specified scale
    
    Args:
        img1 (torch.Tensor): First image batch
        img2 (torch.Tensor): Second image batch
        window (torch.Tensor): Gaussian kernel
        window_size (int): Window size
        val_range (float): Value range
        size_average (bool): Size average flag
        
    Returns:
        tuple: (SSIM value, Contrast sensitivity)
    """
    import torch
    import torch.nn.functional as F
    
    device = img1.device
    
    # Check minimum size required for calculation
    min_size = min(img1.size(-2), img1.size(-1))
    if min_size < window_size:
        # Use a smaller window if image is too small
        new_window_size = min_size
        # Regenerate Gaussian kernel with new size
        sigma = 1.5
        _1D_window = torch.Tensor([np.exp(-(x - new_window_size//2)**2/(2*sigma**2)) 
                                for x in range(new_window_size)]).to(device)
        _1D_window = _1D_window / _1D_window.sum()
        _2D_window = _1D_window.unsqueeze(1) @ _1D_window.unsqueeze(0)
        window = _2D_window.expand(img1.size(1), 1, new_window_size, new_window_size).contiguous()
        window_size = new_window_size
    
    # Compute means
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.shape[1])
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.shape[1]) - mu1_mu2
    
    # Add small constants for numerical stability
    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2
    
    # Calculate SSIM
    num_ssim = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denom_ssim = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num_ssim / denom_ssim
    
    # Calculate contrast sensitivity
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    
    # Apply size averaging if needed
    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(1).mean(1).mean(1)
        cs = cs_map.mean(1).mean(1).mean(1)
    
    return ssim_val, cs


def bpp(logits, indices, image_shape):
    """
    Calculate the bits per pixel (BPP) for a given quantized representation

    This function estimates the actual compression ratio by calculating
    the theoretical minimum bit-rate using the entropy model outputs.

    Args:
        logits (torch.Tensor): Predicted probability distribution from entropy model
        indices (torch.Tensor): Quantized indices  
        image_shape (tuple): Original image shape [B, C, H, W]

    Returns:
        float: Bits per pixel (BPP)
    """
    batch_size, orig_channels, height, width = image_shape
    num_pixels = batch_size * orig_channels * height * width

    # Flatten logits & indices for cross_entropy, handling either 4D or 3D logits:
    if logits.ndim == 4:
        # [B, C, H, W] → [B*H*W, C]
        B, C, H, W = logits.shape
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
    elif logits.ndim == 3:
        # [B, N, C] → [B*N, C]
        B, N, C = logits.shape
        logits_flat = logits.reshape(-1, C)
    else:
        raise ValueError(f"Unexpected logits.ndim={logits.ndim}, expected 3 or 4.")

    # Flatten targets in the same overall order:
    indices_flat = indices.reshape(-1)

    # Compute cross‑entropy in bits:
    cross_entropy = F.cross_entropy(logits_flat, indices_flat, reduction='sum') / math.log(2)

    # Calculate BPP by dividing by the number of pixels:
    bpp_value = cross_entropy / num_pixels

    return bpp_value.item()
