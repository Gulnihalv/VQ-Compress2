import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class OptimizedSelfAttention(nn.Module):
    """
    Memory-efficient Self Attention with dynamic positional encoding
    """
    def __init__(self, channels, num_heads=4, reduction_factor=4, dropout=0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.reduction_factor = reduction_factor
        
        # Head dimension kontrolü
        assert channels % num_heads == 0, f"Channels {channels} must be divisible by num_heads {num_heads}"
        self.head_dim = channels // num_heads
        
        # Spatial reduction (daha agresif - memory için)
        if reduction_factor > 1:
            self.spatial_reduce = nn.AvgPool2d(kernel_size=reduction_factor, stride=reduction_factor)
            # Bu satırı kaldırıyoruz - manuel interpolation kullanacağız
            # self.spatial_restore = nn.Upsample(scale_factor=reduction_factor, mode='bilinear', align_corners=False)
        else:
            self.spatial_reduce = None
        
        # Lightweight projections
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_drop = nn.Dropout(dropout)
        
        # Learnable scaling parameter (başlangıçta çok küçük)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Layer normalization (GroupNorm daha memory efficient)
        self.norm = nn.GroupNorm(num_groups=min(8, channels//8), num_channels=channels)
        
        # Temperature parameter for attention
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
    def _create_positional_encoding(self, h, w, channels, device):
        # Sinusoidal positional encoding
        pe = torch.zeros(1, channels, h, w, device=device)
        
        # Y pozisyonu için encoding
        y_pos = torch.arange(h, device=device, dtype=torch.float32).unsqueeze(1).repeat(1, w)
        # X pozisyonu için encoding  
        x_pos = torch.arange(w, device=device, dtype=torch.float32).unsqueeze(0).repeat(h, 1)
        
        # Her pozisyon tipi için ayrı ayrı encoding hesapla
        # Daha basit ve güvenli yaklaşım
        
        # Pozisyon encoding için dimension sayısı
        pos_dim = channels // 4  # Her tür pozisyon için eşit pay
        if pos_dim == 0:
            pos_dim = 1
            
        # Y pozisyonu için sin/cos encoding
        for i in range(0, pos_dim):
            if i * 2 < channels:
                div_term = 1.0 / (10000.0 ** (2.0 * i / channels))
                pe[0, i*2, :, :] = torch.sin(y_pos * div_term)
            if i * 2 + 1 < channels:
                div_term = 1.0 / (10000.0 ** (2.0 * i / channels))
                pe[0, i*2 + 1, :, :] = torch.cos(y_pos * div_term)
        
        # X pozisyonu için sin/cos encoding
        offset = pos_dim * 2
        for i in range(0, pos_dim):
            if offset + i*2 < channels:
                div_term = 1.0 / (10000.0 ** (2.0 * i / channels))
                pe[0, offset + i*2, :, :] = torch.sin(x_pos * div_term)
            if offset + i*2 + 1 < channels:
                div_term = 1.0 / (10000.0 ** (2.0 * i / channels))
                pe[0, offset + i*2 + 1, :, :] = torch.cos(x_pos * div_term)
        
        return pe * 0.01  # Küçük scale factor
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Skip attention for very small spatial dimensions
        if H * W < 64:  # 8x8'den küçükse skip et
            return x
            
        # Normalization
        identity = x
        x = self.norm(x)
        
        # Orijinal boyutları kaydet
        original_h, original_w = H, W
        
        # Spatial reduction if needed
        if self.spatial_reduce is not None and H > 32 and W > 32:
            x_reduced = self.spatial_reduce(x)
            _, _, H_r, W_r = x_reduced.shape
        else:
            x_reduced = x
            H_r, W_r = H, W
        
        # Dynamic positional encoding
        pos_emb = self._create_positional_encoding(H_r, W_r, C, x.device)
        x_reduced = x_reduced + pos_emb
        
        # Compute Q, K, V in one go
        qkv = self.qkv(x_reduced).chunk(3, dim=1)  # Split into Q, K, V
        q, k, v = qkv
        
        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, H_r * W_r)  # B, heads, head_dim, HW
        k = k.view(B, self.num_heads, self.head_dim, H_r * W_r)  # B, heads, head_dim, HW
        v = v.view(B, self.num_heads, self.head_dim, H_r * W_r)  # B, heads, head_dim, HW
        
        # Efficient attention computation
        # Scale queries
        q = q * (self.head_dim ** -0.5)
        
        # Apply temperature
        q = q * self.temperature
        
        # Compute attention with memory-efficient implementation
        if H_r * W_r > 1024:  # Large spatial size için chunked attention
            attn_output = self._chunked_attention(q, k, v, H_r, W_r)
        else:
            # Standard attention for smaller sizes
            attn = torch.matmul(q.transpose(-2, -1), k)  # B, heads, HW, HW
            attn = F.softmax(attn, dim=-1)
            attn_output = torch.matmul(attn, v.transpose(-2, -1))  # B, heads, HW, head_dim
            attn_output = attn_output.transpose(-2, -1)  # B, heads, head_dim, HW
        
        # Reshape back to spatial format
        attn_output = attn_output.contiguous().view(B, C, H_r, W_r)
        
        # Restore spatial dimensions if reduced - FIXED VERSION
        if H_r != original_h or W_r != original_w:
            # Manuel interpolation ile boyutları tam olarak eşleştir
            attn_output = F.interpolate(
                attn_output, 
                size=(original_h, original_w), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Output projection
        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)
        
        # Residual connection with learnable scaling
        return identity + self.gamma * attn_output
    
    def _chunked_attention(self, q, k, v, h, w):
        B, num_heads, head_dim, spatial_size = q.shape
        chunk_size = min(256, spatial_size)  # Adaptive chunk size
        
        # Pre-allocate output tensor
        attn_output = torch.zeros_like(v)
        
        # Process in chunks to save memory
        for i in range(0, spatial_size, chunk_size):
            end_i = min(i + chunk_size, spatial_size)
            q_chunk = q[:, :, :, i:end_i]  # B, heads, head_dim, chunk
            
            # Compute attention for this chunk
            attn_chunk = torch.matmul(q_chunk.transpose(-2, -1), k)  # B, heads, chunk, spatial_size
            attn_chunk = F.softmax(attn_chunk, dim=-1)
            
            # Apply attention
            out_chunk = torch.matmul(attn_chunk, v.transpose(-2, -1))  # B, heads, chunk, head_dim
            attn_output[:, :, :, i:end_i] = out_chunk.transpose(-2, -1)
            
            # Clear intermediate tensors to save memory
            del attn_chunk, out_chunk
        
        return attn_output


class LightweightChannelAttention(nn.Module):
    """
    Lightweight channel attention - spatial attention'a alternatif
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Daha küçük bottleneck
        mid_channels = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class EfficientResidualBlock(nn.Module):
    """
    Memory-efficient residual block with optional attention
    """
    def __init__(self, channels, use_attention=True, attention_type="channel"):
        super().__init__()
        
        # Depthwise separable convolution for efficiency
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        )
        self.norm1 = nn.GroupNorm(num_groups=min(8, channels//8), num_channels=channels)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        )
        self.norm2 = nn.GroupNorm(num_groups=min(8, channels//8), num_channels=channels)
        
        # Attention selection - sadece channel attention kullan (memory için)
        if use_attention and attention_type == "channel":
            self.attention = LightweightChannelAttention(channels)
        elif use_attention and attention_type == "spatial":
            # Sadece küçük spatial size'larda spatial attention kullan
            self.attention = OptimizedSelfAttention(channels, num_heads=2, reduction_factor=4)
        else:
            self.attention = None
            
        # Activation
        self.activation = nn.GELU()
        
    def forward(self, x):
        identity = x
        
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        
        # Attention uygula
        if self.attention is not None:
            out = self.attention(out)
        
        out += identity
        return self.activation(out)


# Backward compatibility için
class SelfAttention(OptimizedSelfAttention):
    """Backward compatibility wrapper"""
    pass

class ChannelAttention(LightweightChannelAttention):
    """Backward compatibility wrapper"""
    pass

class ResidualBlock(EfficientResidualBlock):
    """Backward compatibility wrapper"""
    pass