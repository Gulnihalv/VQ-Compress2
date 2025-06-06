import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.helper_models import ResidualBlock, SelfAttention

class Decoder(nn.Module):
    """
    Skip connection'lar olmadığında da çalışabilen decoder
    """
    def __init__(self, out_channels=3, latent_channels=192, num_layers=3, 
                 post_vq_layers=0, downsampling_rate=8, use_attention=True,
                 encoder_channels=None):
        super().__init__()
        
        # Calculate number of upsampling operations needed
        if downsampling_rate in [8, 16]:
            self.us_num_layers = 3 if downsampling_rate == 8 else 4
        else:
            raise ValueError(f"Downsampling rate {downsampling_rate} not supported. Use 8 or 16.")
            
        self.downsampling_rate = downsampling_rate
        self.use_attention = use_attention
        
        # Post VQ processing layers
        self.post_vq = nn.ModuleList()
        
        for _ in range(post_vq_layers):
            self.post_vq.append(
                nn.Sequential(
                    ResidualBlock(latent_channels),
                    nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(latent_channels),
                    nn.ReLU(True)
                )
            )
        
        # Skip connection setup
        if encoder_channels is None:
            print("INFO: encoder_channels not provided, using standalone mode")
            self.use_skip_connections = False
            encoder_channels = []
        else:
            self.use_skip_connections = True
            print(f"INFO: Skip connections enabled with channels: {encoder_channels}")
        
        # Base decoder channels (skip connection olmadığında kullanılacak)
        self.base_decoder_channels = [latent_channels]
        for i in range(self.us_num_layers):
            base_ch = max(self.base_decoder_channels[-1] // 2, latent_channels // (2 ** self.us_num_layers))
            self.base_decoder_channels.append(base_ch)
        
        # Skip connection varsa effective channels hesapla
        if self.use_skip_connections:
            self.effective_channels = []
            for i in range(self.us_num_layers):
                base_ch = self.base_decoder_channels[i]
                if i < len(encoder_channels):
                    skip_ch = encoder_channels[-(i+1)]  # Reverse order
                    effective_in_ch = base_ch + skip_ch
                else:
                    effective_in_ch = base_ch
                self.effective_channels.append(effective_in_ch)
        else:
            self.effective_channels = self.base_decoder_channels[:-1]  # Son hariç
        
        print(f"Base decoder channels: {self.base_decoder_channels}")
        print(f"Effective input channels: {self.effective_channels}")
        
        # Upsampling layers - hem skip connectionlı hem de standalone çalışabilir
        self.upsample_layers = nn.ModuleList()
        for i in range(self.us_num_layers):
            # Maximum possible input channels (skip connection varsa)
            max_in_ch = self.effective_channels[i]
            out_ch = self.base_decoder_channels[i+1]
            
            # Flexible input layer - farklı channel sayılarını handle edebilir
            up_block = nn.ModuleDict({
                'pre_process': nn.Sequential(
                    nn.Conv2d(max_in_ch, out_ch, kernel_size=1),  # Channel adaptation
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(True)
                ),
                'main_process': nn.Sequential(
                    ResidualBlock(out_ch),
                    nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(True),
                    ResidualBlock(out_ch)
                )
            })
            
            # Standalone mode için alternatif path
            up_block['standalone'] = nn.Sequential(
                ResidualBlock(self.base_decoder_channels[i]),
                nn.ConvTranspose2d(self.base_decoder_channels[i], out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
                ResidualBlock(out_ch)
            )
            
            self.upsample_layers.append(up_block)
        
        # Additional processing layers
        num_extra_layers = max(0, num_layers - self.us_num_layers)
        self.extra_layers = nn.ModuleList()
        
        current_ch = self.base_decoder_channels[-1]
        for i in range(num_extra_layers):
            next_ch = max(current_ch // 2, 16)
            
            self.extra_layers.append(
                nn.Sequential(
                    ResidualBlock(current_ch),
                    nn.Conv2d(current_ch, next_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(next_ch),
                    nn.ReLU(True)
                )
            )
            current_ch = next_ch
        
        final_channels = current_ch
        
        # Final attention
        if use_attention:
            self.final_attention = SelfAttention(final_channels)
        else:
            self.final_attention = None
            
        # Final output layer
        self.final = nn.Sequential(
            nn.Conv2d(final_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def _adapt_spatial_size(self, tensor, target_spatial_shape):
        """Tensörün mekansal boyutunu hedef boyuta uyarlar."""
        if tensor.shape[2:] != target_spatial_shape:
            tensor = F.interpolate(
                tensor, 
                size=target_spatial_shape, 
                mode='bilinear', 
                align_corners=False
            )
        return tensor
        
    def forward(self, z, encoder_features=None):
        # Post-VQ processing
        for layer in self.post_vq:
            z = layer(z)
        
        # Skip connection'ların mevcut olup olmadığını kontrol et
        use_skip = (self.use_skip_connections and 
                   encoder_features is not None and 
                   len(encoder_features) > 0)
        
        # Upsampling layers
        for idx, upsample_layer in enumerate(self.upsample_layers):
            
            if use_skip and idx < len(encoder_features):
                # Skip connection modu
                enc_feat = encoder_features[-(idx+1)]
                enc_feat = self._adapt_spatial_size(enc_feat, z.shape[2:])
                
                # Concatenate skip features
                z = torch.cat([z, enc_feat], dim=1)
                
                # Pre-process ile channel'ı uyarla
                z = upsample_layer['pre_process'](z)
                z = upsample_layer['main_process'](z)
                
            else:
                # Standalone modu - skip connection yok
                z = upsample_layer['standalone'](z)
        
        # Extra processing layers
        for layer in self.extra_layers:
            z = layer(z)
        
        # Final attention
        if self.final_attention is not None:
            z = self.final_attention(z)
        
        # Final output
        out = self.final(z)
        return out