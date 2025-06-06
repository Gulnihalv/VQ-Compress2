import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average updates.
    
    Bu implementasyon index corruption sorunlarını önlemek için
    güvenlik kontrolleri içerir.
    """
    def __init__(self, num_embeddings=1024, embedding_dim=128, commitment_coef=0.25, decay=0.99, debug=False):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_coef = commitment_coef
        self.decay = decay
        self.debug = debug
        
        # Initialize codebook as a nn.Embedding layer
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embed.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)
        
        # EMA related buffers
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embed.weight.data.clone())
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        
        # Debug ve monitoring için
        self.register_buffer('corruption_count', torch.tensor(0, dtype=torch.long))
        
    def _validate_indices(self, indices, context="unknown"):
        """Index validasyonu ve debugging"""
        if indices.numel() == 0:
            return indices
            
        min_idx = indices.min().item()
        max_idx = indices.max().item()
        
        if min_idx < 0 or max_idx >= self.num_embeddings:
            corruption_detected = True
            self.corruption_count += 1
            
            if self.debug:
                logging.warning(f"[{context}] Index corruption detected:")
                logging.warning(f"  - Index range: [{min_idx}, {max_idx}]")
                logging.warning(f"  - Valid range: [0, {self.num_embeddings-1}]")
                logging.warning(f"  - Total corruptions: {self.corruption_count}")
                
                # Histgram analizi
                unique, counts = torch.unique(indices, return_counts=True)
                invalid_mask = (unique < 0) | (unique >= self.num_embeddings)
                if invalid_mask.any():
                    invalid_indices = unique[invalid_mask]
                    invalid_counts = counts[invalid_mask]
                    logging.warning(f"  - Invalid indices: {invalid_indices.tolist()}")
                    logging.warning(f"  - Invalid counts: {invalid_counts.tolist()}")
            
            # Güvenli clamp
            indices = torch.clamp(indices, 0, self.num_embeddings - 1)
            
        return indices
    
    def _safe_distance_calculation(self, flat_input):
        """Numerically stable distance calculation"""
        # L2 distance hesaplama - numerical stability için
        input_norm = torch.sum(flat_input**2, dim=1, keepdim=True)
        embed_norm = torch.sum(self.embed.weight**2, dim=1)
        
        # Matmul için numerical stability check
        dot_product = torch.matmul(flat_input, self.embed.weight.t())
        
        # NaN/Inf kontrolü
        if torch.isnan(dot_product).any() or torch.isinf(dot_product).any():
            logging.error("NaN/Inf detected in dot product calculation!")
            # Fallback: sadece L2 norm kullan
            distances = input_norm + embed_norm.unsqueeze(0)
        else:
            distances = input_norm + embed_norm - 2 * dot_product
        
        # Negatif distance'ları temizle (numerical error)
        distances = torch.clamp(distances, min=0.0)
        
        return distances

    def forward(self, z_e):
        """
        Quantize the input tensor and compute the quantization loss.
        """
        # Input validation
        if torch.isnan(z_e).any() or torch.isinf(z_e).any():
            logging.error("NaN/Inf detected in input tensor!")
            z_e = torch.nan_to_num(z_e, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Reshape input for distance calculation
        input_shape = z_e.shape
        flat_input = z_e.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        
        # Safe distance calculation
        distances = self._safe_distance_calculation(flat_input)
        
        # Encoding with validation
        encoding_indices = torch.argmin(distances, dim=1)
        encoding_indices = self._validate_indices(encoding_indices, "forward")
        
        # One-hot encoding
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Quantize
        quantized = self.embed(encoding_indices).view(input_shape[0], input_shape[2], input_shape[3], self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        # EMA update (only during training)
        if self.training:
            try:
                self._ema_update(encodings, flat_input)
            except Exception as e:
                logging.error(f"EMA update failed: {e}")
                # EMA update başarısız olursa training'i durdurma
        
        # Compute loss
        commitment_loss = F.mse_loss(quantized.detach(), z_e)
        
        # Straight-through estimator
        quantized = z_e + (quantized - z_e).detach()
        
        # Reshape indices for batch format
        indices = encoding_indices.view(input_shape[0], input_shape[2] * input_shape[3])
        
        return quantized, indices, commitment_loss * self.commitment_coef
    
    def _ema_update(self, encodings, flat_input):
        """
        Safe EMA update with numerical stability checks
        """
        # Input validation
        if torch.isnan(encodings).any() or torch.isnan(flat_input).any():
            logging.warning("NaN detected in EMA update inputs, skipping update")
            return
        
        # EMA update for embedding vectors
        self.cluster_size = encodings.sum(0)
        
        # Numerical stability check
        n = self.cluster_size.sum()
        if n == 0:
            logging.warning("Empty cluster detected, skipping EMA update")
            return
        
        # Update ema_cluster_size
        self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * self.cluster_size
        
        # Compute embeddings from the current batch
        dw = torch.matmul(encodings.t(), flat_input)
        
        # Update ema_w
        self.ema_w = self.decay * self.ema_w + (1 - self.decay) * dw
        
        # Normalize cluster sizes with numerical stability
        n_ema = self.ema_cluster_size.sum()
        if n_ema > 0:
            cluster_size = ((self.ema_cluster_size + 1e-5) / (n_ema + self.num_embeddings * 1e-5) * n_ema)
            
            # Division by zero kontrolü
            cluster_size = torch.clamp(cluster_size, min=1e-8)
            
            # Update embeddings
            normalized_ema_w = self.ema_w / cluster_size.unsqueeze(1)
            
            # NaN kontrolü
            if not torch.isnan(normalized_ema_w).any():
                self.embed.weight.data.copy_(normalized_ema_w)
            else:
                logging.warning("NaN detected in embedding update, skipping")
        
    def quantize(self, z_e):
        """
        Safe quantization without loss computation
        """
        # Input validation
        if torch.isnan(z_e).any() or torch.isinf(z_e).any():
            logging.warning("NaN/Inf in quantize input, cleaning...")
            z_e = torch.nan_to_num(z_e, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Reshape input
        input_shape = z_e.shape
        flat_input = z_e.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        
        # Safe distance calculation
        distances = self._safe_distance_calculation(flat_input)
        
        # Get indices with validation
        encoding_indices = torch.argmin(distances, dim=1)
        encoding_indices = self._validate_indices(encoding_indices, "quantize")
        
        # Quantize
        quantized = self.embed(encoding_indices).view(input_shape[0], input_shape[2], input_shape[3], self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        # Reshape indices
        indices = encoding_indices.view(input_shape[0], input_shape[2] * input_shape[3])
        
        return quantized, indices
    
    def lookup_indices(self, indices, spatial_shape):
        """
        Güvenli index lookup with comprehensive validation
        """
        batch_size, h, w = spatial_shape
        
        # Input validation
        if indices is None or indices.numel() == 0:
            raise ValueError("Empty indices tensor provided")
        
        # Spatial shape validation
        expected_spatial_size = h * w
        actual_spatial_size = indices.size(1)
        if expected_spatial_size != actual_spatial_size:
            raise ValueError(f"Spatial shape mismatch: expected {expected_spatial_size}, got {actual_spatial_size}")
        
        # Index validation ve cleaning
        indices = self._validate_indices(indices, "lookup_indices")
        
        # Flatten indices and lookup embeddings
        flat_indices = indices.view(-1)
        embeddings = self.embed(flat_indices)
        
        # Reshape to spatial format
        z_q = embeddings.view(batch_size, h, w, self.embedding_dim)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # [B, embedding_dim, H, W]
        
        return z_q
    
    def get_codebook_usage(self):
        """Enhanced codebook usage statistics"""
        with torch.no_grad():
            total_size = self.ema_cluster_size.sum()
            if total_size > 0:
                # Calculate usage ratio
                used_codes = (self.ema_cluster_size > 1e-7).sum().item()
                usage_ratio = used_codes / self.num_embeddings
                
                # Calculate entropy
                probs = self.ema_cluster_size / total_size
                probs = probs[probs > 1e-7]
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()  # Numerical stability
                max_entropy = torch.log(torch.tensor(self.num_embeddings, dtype=torch.float)).item()
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
                
                return {
                    'usage_ratio': usage_ratio,
                    'used_codes': used_codes,
                    'total_codes': self.num_embeddings,
                    'entropy': entropy,
                    'normalized_entropy': normalized_entropy,
                    'cluster_sizes': self.ema_cluster_size.cpu().numpy(),
                    'corruption_count': self.corruption_count.item(),
                    'max_cluster_size': self.ema_cluster_size.max().item(),
                    'min_cluster_size': self.ema_cluster_size.min().item()
                }
            else:
                return {
                    'usage_ratio': 0.0,
                    'used_codes': 0,
                    'total_codes': self.num_embeddings,
                    'entropy': 0.0,
                    'normalized_entropy': 0.0,
                    'cluster_sizes': self.ema_cluster_size.cpu().numpy(),
                    'corruption_count': self.corruption_count.item(),
                    'max_cluster_size': 0.0,
                    'min_cluster_size': 0.0
                }
    
    def reset_corruption_counter(self):
        """Corruption counter'ı sıfırla"""
        self.corruption_count.zero_()
    
    def get_health_status(self):
        """VQ modülünün sağlık durumunu kontrol et"""
        stats = self.get_codebook_usage()
        
        issues = []
        if stats['corruption_count'] > 0:
            issues.append(f"Index corruption detected: {stats['corruption_count']} times")
        
        if stats['usage_ratio'] < 0.1:
            issues.append(f"Low codebook usage: {stats['usage_ratio']:.2%}")
        
        if stats['normalized_entropy'] < 0.5:
            issues.append(f"Low entropy: {stats['normalized_entropy']:.3f}")
        
        # NaN kontrolleri
        if torch.isnan(self.embed.weight).any():
            issues.append("NaN detected in embedding weights")
        
        if torch.isnan(self.ema_cluster_size).any():
            issues.append("NaN detected in EMA cluster sizes")
        
        return {
            'healthy': len(issues) == 0,
            'issues': issues,
            'stats': stats
        }