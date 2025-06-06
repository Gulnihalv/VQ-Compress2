import torch.nn as nn
from utils.helper_models import ResidualBlock, SelfAttention

class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_channels=128, num_layers=3, downsampling_rate=8, use_attention=True):
        super().__init__()
        
        # Calculate number of downsampling operations needed
        if downsampling_rate in [8, 16]:
            ds_num_layers = 3 if downsampling_rate == 8 else 4
        else:
            raise ValueError(f"Downsampling rate {downsampling_rate} not supported. Use 8 or 16.")
            
        self.downsampling_rate = downsampling_rate
        self.use_attention = use_attention
        self.ds_num_layers = ds_num_layers
        
        # Initial convolutional layer
        base_channels = latent_channels // 4
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True)
        )
        
        # Downsampling layers with progressive channel expansion
        self.downsample_layers = nn.ModuleList()
        
        # Channel dimensions for each layer
        self.channel_progression = [base_channels]
        for i in range(ds_num_layers):
            next_ch = min(self.channel_progression[-1] * 2, latent_channels)
            self.channel_progression.append(next_ch)
        
        # Create downsampling blocks
        for i in range(ds_num_layers):
            in_ch = self.channel_progression[i]
            out_ch = self.channel_progression[i+1]
            
            down_block = nn.Sequential(
                ResidualBlock(in_ch),
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
                ResidualBlock(out_ch)
            )
            
            self.downsample_layers.append(down_block)
        
        # Additional processing layers without downsampling
        num_extra_layers = max(0, num_layers - ds_num_layers)
        self.extra_layers = nn.ModuleList()
        
        for _ in range(num_extra_layers):
            self.extra_layers.append(
                nn.Sequential(
                    ResidualBlock(self.channel_progression[-1]),
                    nn.Conv2d(self.channel_progression[-1], self.channel_progression[-1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(self.channel_progression[-1]),
                    nn.ReLU(True)
                )
            )
        
        # Only final attention (if enabled)
        if use_attention:
            self.final_attention = SelfAttention(self.channel_progression[-1])
        else:
            self.final_attention = None
        
        # Store final output channels for easy access
        self._output_channels = self.channel_progression[-1]
            
    def forward(self, x):
        x = self.initial(x)
        
        # Store features specifically for skip connections (only downsampling outputs)
        skip_features = []
        
        # Apply downsampling layers and store their outputs
        for i, layer in enumerate(self.downsample_layers):
            x = layer(x)
            skip_features.append(x)  # Bu decoder için gerekli olan feature'lar
        
        # Apply extra processing layers (bunlar skip connection'da kullanılmayacak)
        for layer in self.extra_layers:
            x = layer(x)
        
        # Apply final attention only
        if self.final_attention is not None:
            x = self.final_attention(x)
            
        return x, skip_features  # Sadece downsampling çıktılarını döndür
    
    @property
    def output_channels(self):
        """Returns the number of output channels from the encoder"""
        return self._output_channels
    
    def get_downsampling_channels(self):
        """Returns the channel progression for downsampling layers (decoder için gerekli)"""
        return self.channel_progression[1:]  # İlk initial channels'ı hariç tut
    
    def get_output_shape(self, input_shape):
        """
        Calculate the output shape given an input shape
        Args:
            input_shape: tuple (C, H, W) or (B, C, H, W)
        Returns:
            tuple: Output shape
        """
        if len(input_shape) == 4:
            batch_size, channels, height, width = input_shape
        else:
            channels, height, width = input_shape
            batch_size = None
        
        # Calculate spatial dimensions after downsampling
        output_height = height // self.downsampling_rate
        output_width = width // self.downsampling_rate
        
        if batch_size is not None:
            return (batch_size, self.output_channels, output_height, output_width)
        else:
            return (self.output_channels, output_height, output_width)