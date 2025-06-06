import torch
import torch.nn as nn
import torch.nn.functional as F

class EntropyModel(nn.Module):
    def __init__(self, num_embeddings=512, num_layers=2, hidden_dim=256, num_heads=4):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.token_emb = nn.Embedding(num_embeddings, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to_logits = nn.Linear(hidden_dim, num_embeddings)

    def forward(self, indices):
        # indices: (B, N) where N = H*W
        x = self.token_emb(indices)  # (B, N, hidden_dim)
        # Permute for transformer: (N, B, hidden_dim)
        x = x.permute(1,0,2)
        t_out = self.transformer(x)
        logits = self.to_logits(t_out)  # (N, B, num_embeddings)
        logits = logits.permute(1,0,2)  # (B, N, K)
        return logits
    
    def get_probabilities(self, indices):
        """İndeksler için olasılık dağılımlarını hesapla"""
        with torch.no_grad():
            logits = self.forward(indices)
            probs = F.softmax(logits, dim=-1)
        return probs