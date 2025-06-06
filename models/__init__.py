import torch
import torch.nn as nn

class VQCompressionModel(nn.Module):
    def __init__(self, encoder, vq, decoder, entropy_model):
        super().__init__()
        self.encoder = encoder
        
        # Encoder output channels'ı al
        encoder_output_dim = getattr(encoder, '_output_channels', encoder.output_channels) 
        
        self.pre_vq = nn.Conv2d(encoder_output_dim, 
                               vq.embedding_dim, kernel_size=1)
        self.vq = vq
        self.post_vq = nn.Conv2d(vq.embedding_dim, vq.embedding_dim, kernel_size=1)
        
        self.decoder = decoder
        self.entropy_model = entropy_model
        
        # Spatial shape ve features cache
        self._spatial_shape_cache = {}
        self._encoder_features_cache = {}
        
        # Decoder'ın skip connection'ları destekleyip desteklemediğini kontrol et
        self.use_skip_connections = getattr(decoder, 'use_skip_connections', False)
        
        print(f"VQCompressionModel initialized:")
        print(f"  Encoder output channels: {encoder_output_dim}")
        print(f"  VQ embedding dim: {vq.embedding_dim}")
        print(f"  Skip connections enabled: {self.use_skip_connections}")
        
    def forward(self, x):
        batch_size, _, height, width = x.shape
        encoded_height = height // self.encoder.downsampling_rate
        encoded_width = width // self.encoder.downsampling_rate
        
        # Encoder'dan hem latent hem de skip features al
        z, skip_features = self.encoder(x)
        
        # VQ processing
        z = self.pre_vq(z)
        z_q, indices, vq_loss = self.vq(z)
        z_q = self.post_vq(z_q)
        
        # Decoder'ı skip features ile çağır
        if self.use_skip_connections:
            recon = self.decoder(z_q, skip_features)
        else:
            recon = self.decoder(z_q)
        
        # Entropy model
        logits = self.entropy_model(indices)   
        
        return recon, indices, logits, vq_loss
    
    def encode(self, x):
        """
        Encode input to indices and cache encoder features for decoding
        """
        batch_size, _, height, width = x.shape
        encoded_height = height // self.encoder.downsampling_rate
        encoded_width = width // self.encoder.downsampling_rate
        
        # Spatial shape'i cache'le
        self._spatial_shape_cache[batch_size] = (encoded_height, encoded_width)
        
        # Encoder'dan hem latent hem de skip features al
        z, skip_features = self.encoder(x)
        
        # Skip features'ı cache'le (decoding için gerekli)
        if self.use_skip_connections:
            self._encoder_features_cache[batch_size] = skip_features
        
        # VQ processing
        z = self.pre_vq(z)
        _, indices, _ = self.vq(z)
        
        return indices
    
    def decode(self, indices, spatial_shape=None, encoder_features=None):
        """
        Decode indices to reconstruction
        
        Args:
            indices: VQ indices
            spatial_shape: Optional spatial shape (H, W)
            encoder_features: Optional encoder features for skip connections
        """
        batch_size = indices.size(0)
        spatial_size = indices.size(1)
        
        # Spatial shape'i belirle
        if spatial_shape is not None:
            h, w = spatial_shape
            if h * w != spatial_size:
                raise ValueError(f"Spatial shape {spatial_shape} doesn't match indices size {spatial_size}")
        else:
            if batch_size in self._spatial_shape_cache:
                h, w = self._spatial_shape_cache[batch_size]
            else:
                h = w = int(spatial_size ** 0.5)
                if h * w != spatial_size:
                    raise ValueError(f"Cannot infer spatial shape from indices size {spatial_size}. "
                                   f"Please provide spatial_shape parameter.")
        
        # Encoder features'ı belirle
        if self.use_skip_connections:
            if encoder_features is not None:
                skip_features = encoder_features
            elif batch_size in self._encoder_features_cache:
                skip_features = self._encoder_features_cache[batch_size]
            else:
                print("WARNING: Skip connections enabled but no encoder features available. "
                      "Decoding without skip connections.")
                skip_features = None
        else:
            skip_features = None
        
        # VQ lookup
        z_q = self.vq.lookup_indices(indices, (batch_size, h, w))
        z_q = self.post_vq(z_q)
        
        # Decoder'ı çağır
        if skip_features is not None:
            recon = self.decoder(z_q, skip_features)
        else:
            recon = self.decoder(z_q)
        
        return recon
    
    def clear_cache(self):
        """Clear cached spatial shapes and encoder features"""
        self._spatial_shape_cache.clear()
        self._encoder_features_cache.clear()
    
    def get_compression_ratio(self, input_shape):
        batch_size, channels, height, width = input_shape
        original_bits = channels * height * width * 8
        encoded_height = height // self.encoder.downsampling_rate
        encoded_width = width // self.encoder.downsampling_rate
        compressed_bits = encoded_height * encoded_width * torch.log2(torch.tensor(self.vq.num_embeddings)).item()     
        return original_bits / compressed_bits
    
    def encode_decode(self, x):
        """
        Convenience method for full encode-decode cycle
        Maintains encoder features for skip connections
        """
        # Encode
        indices = self.encode(x)
        
        # Decode (features are cached from encode step)
        recon = self.decode(indices)
        
        return recon, indices