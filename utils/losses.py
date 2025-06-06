import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG19_Weights
import numpy as np

class PerceptualLoss(nn.Module):
    """
    VGG tabanlı perceptual loss - görsel kaliteyi artırmak için kritik
    """
    def __init__(self, layers=['relu_1_1', 'relu_2_1', 'relu_3_1', 'relu_4_1'], 
                 weights=[1.0, 1.0, 1.0, 1.0]):
        super().__init__()
        
        # Pre-trained VGG19 model
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.vgg = vgg.eval()
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        self.layer_name_mapping = {
            '3': "relu_1_1",   # After first ReLU
            '8': "relu_2_1",   # After first ReLU in block 2
            '17': "relu_3_1",  # After first ReLU in block 3
            '26': "relu_4_1"   # After first ReLU in block 4
        }
        
        self.target_layers = layers
        self.weights = weights
        
        # Normalization values for ImageNet
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def normalize_input(self, x):
        """Normalize input to match ImageNet preprocessing"""
        return (x - self.mean) / self.std
    
    def forward(self, pred, target):
        """
        Compute perceptual loss between prediction and target
        """
        # Normalize inputs
        pred_norm = self.normalize_input(pred)
        target_norm = self.normalize_input(target)
        
        # Extract features
        pred_features = self.extract_features(pred_norm)
        target_features = self.extract_features(target_norm)
        
        # Compute weighted loss
        loss = 0.0
        for i, layer in enumerate(self.target_layers):
            if layer in pred_features and layer in target_features:
                weight = self.weights[i] if i < len(self.weights) else 1.0
                feature_loss = F.mse_loss(pred_features[layer], target_features[layer])
                loss += weight * feature_loss
                
        return loss
    
    def extract_features(self, x):
        """Extract features from specified VGG layers"""
        features = {}
        
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layer_name_mapping:
                layer_name = self.layer_name_mapping[name]
                if layer_name in self.target_layers:
                    features[layer_name] = x
                    
        return features


class MS_SSIM_Loss(nn.Module):
    """
    Multi-Scale Structural Similarity Index loss
    """
    def __init__(self, data_range=1.0, size_average=True, channel=3, 
                 weights=[0.4, 0.4, 0.2]):  # 3 scale için optimize edildi
        super().__init__()
        self.data_range = data_range
        self.size_average = size_average
        self.channel = channel
        self.weights = weights
        self.register_buffer('window', self._create_window(11, channel))

        
    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def _gaussian_window(self, window_size, sigma):
        # Modülün bağlı olduğu cihazı al
        # Modülün en az bir parametresi veya tamponu olduğunu varsayarız.
        # Genellikle, torch.nn.Module'ler .to(device) ile taşındığında bu bilgiye erişilebilir.
        if hasattr(self, 'window') and self.window is not None:
            module_device = self.window.device
        elif len(list(self.buffers())) > 0:
            module_device = next(self.buffers()).device
        elif len(list(self.parameters())) > 0:
            module_device = next(self.parameters()).device
        else:
            # Eğer hiç tampon veya parametre yoksa, varsayılan olarak CPU kullan
            module_device = torch.device("cpu")
            print("Uyarı: MS_SSIM_Loss modülünde parametre veya tampon bulunamadı, CPU varsayılan olarak kullanılıyor.")

        # x değerlerini bir Tensor olarak oluştur ve doğru cihaza taşı
        x = torch.arange(window_size, dtype=torch.float32, device=module_device)
        
        # Tüm hesaplamaları Tensor operasyonları olarak yapın
        gauss = torch.exp(-((x - window_size//2)**2) / (2 * sigma**2))
        
        # Normalizasyon
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        channel = img1.size(1)
        device = img1.device
        
        ms_ssim_values = []
        
        for i, weight in enumerate(self.weights):
            if i > 0:
                # Downsample for multi-scale
                img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
                img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            
            # Skip if images become too small
            if min(img1.shape[2:]) < 11:
                break
                
            window_size = 11
            window = self._create_window(window_size, channel).to(device)
            
            ssim_val = self._ssim(img1, img2, window, window_size, channel, self.size_average)
            ms_ssim_values.append(ssim_val * weight)
        
        return 1 - sum(ms_ssim_values)  # Return as loss (1 - SSIM)


class CombinedLoss(nn.Module): # Bu class basitleştirildi
    """
    Basitleştirilmiş combined loss - sadece aktif loss'lar
    """
    def __init__(self, 
                 l1_weight=1.0,
                 l2_weight=0.0,        # Genelde kapalı
                 perceptual_weight=0.3,
                 ms_ssim_weight=0.0,   # Genelde kapalı
                 vq_weight=1.0,
                 entropy_weight=0.01):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.perceptual_weight = perceptual_weight
        self.ms_ssim_weight = ms_ssim_weight
        self.vq_weight = vq_weight
        self.entropy_weight = entropy_weight
        
        # Loss modules - sadece aktif olanlar
        if self.perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss(
                layers=['relu_2_1', 'relu_3_1'],  # Sadece 2 katman
                weights=[0.5, 0.5]
            )
        
        if self.ms_ssim_weight > 0:
            self.ms_ssim_loss = MS_SSIM_Loss(
                weights=[0.4, 0.4, 0.2]  # 3 scale'e düşürüldü
            )
        
    def forward(self, pred, target, vq_loss, entropy_loss):
        """
        Sadece aktif loss'ları hesapla
        """
        losses = {}
        
        # Her zaman aktif olan loss'lar
        l1_loss = F.l1_loss(pred, target)
        losses['l1'] = l1_loss
        
        # L2 sadece ağırlık > 0 ise
        if self.l2_weight > 0:
            l2_loss = F.mse_loss(pred, target)
            losses['l2'] = l2_loss
        else:
            losses['l2'] = torch.tensor(0.0, device=pred.device)
        
        # Perceptual loss - sadece ağırlık > 0 ise
        if self.perceptual_weight > 0 and hasattr(self, 'perceptual_loss'):
            perc_loss = self.perceptual_loss(pred, target)
            losses['perceptual'] = perc_loss
        else:
            losses['perceptual'] = torch.tensor(0.0, device=pred.device)
        
        # MS-SSIM loss - sadece ağırlık > 0 ise
        if self.ms_ssim_weight > 0 and hasattr(self, 'ms_ssim_loss'):
            ssim_loss = self.ms_ssim_loss(pred, target)
            losses['ms_ssim'] = ssim_loss
        else:
            losses['ms_ssim'] = torch.tensor(0.0, device=pred.device)
        
        # VQ ve entropy her zaman aktif
        losses['vq'] = vq_loss
        losses['entropy'] = entropy_loss
        
        # Total loss - sadece aktif olanları topla
        total_loss = (self.l1_weight * l1_loss + 
                     self.vq_weight * vq_loss +
                     self.entropy_weight * entropy_loss)
        
        # Opsiyonel loss'ları ekle
        if self.l2_weight > 0:
            total_loss += self.l2_weight * losses['l2']
        
        if self.perceptual_weight > 0 and hasattr(self, 'perceptual_loss'):
            total_loss += self.perceptual_weight * losses['perceptual']
            
        if self.ms_ssim_weight > 0 and hasattr(self, 'ms_ssim_loss'):
            total_loss += self.ms_ssim_weight * losses['ms_ssim']
        
        losses['total'] = total_loss
        
        return total_loss, losses

class AdaptiveLossWeights(nn.Module):
    """
    Eğitim sırasında loss ağırlıklarını adaptif olarak ayarlayan modül
    """
    def __init__(self, initial_weights, adaptation_rate=0.01):
        super().__init__()
        
        self.adaptation_rate = adaptation_rate
        
        # Learnable loss weights
        for name, weight in initial_weights.items():
            self.register_parameter(f"{name}_weight", 
                                  nn.Parameter(torch.tensor(weight, dtype=torch.float32)))
    
    def forward(self, losses_dict):
        """
        Compute adaptive weighted loss
        """
        total_loss = 0
        weights = {}
        
        for loss_name, loss_value in losses_dict.items():
            if loss_name != 'total':
                weight_param = getattr(self, f"{loss_name}_weight")
                weights[loss_name] = torch.sigmoid(weight_param)  # Ensure positive
                total_loss += weights[loss_name] * loss_value
        
        return total_loss, weights
    
    def update_weights(self, losses_dict, target_ratios):
        """
        Update loss weights based on loss magnitudes
        """
        with torch.no_grad():
            for loss_name, target_ratio in target_ratios.items():
                if loss_name in losses_dict:
                    current_loss = losses_dict[loss_name].item()
                    weight_param = getattr(self, f"{loss_name}_weight")
                    
                    # Simple adaptive update
                    if current_loss > target_ratio:
                        weight_param.data *= (1 - self.adaptation_rate)
                    else:
                        weight_param.data *= (1 + self.adaptation_rate)
                    
                    # Clamp weights to reasonable range
                    weight_param.data.clamp_(0.001, 10.0)


# Özel loss fonksiyonları ve yardımcı metrikler
def psnr(img1, img2):
    """Peak Signal-to-Noise Ratio hesaplama"""
    mse = F.mse_loss(img1, img2)
    if mse < 1e-10:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def calculate_entropy_loss(logits, indices):
    """Entropi kaybını hesaplar - train.py'deki ile uyumlu"""
    B, N = indices.shape
    targets = indices.detach()
    
    # Tensor'u contiguous hale getirme
    logits = logits.contiguous()
    
    # Güvenle view kullanabilmek için
    logits = logits.view(-1, logits.size(-1))
    targets = targets.reshape(-1)
    
    entropy_loss = F.cross_entropy(logits, targets)
    
    return entropy_loss


class WarmupLossScheduler:
    """
    Basitleştirilmiş loss warmup scheduler
    """
    def __init__(self, warmup_epochs=5):  # 10'dan 5'e düşürüldü
        self.warmup_epochs = warmup_epochs
        
    def get_loss_multiplier(self, epoch, loss_type='perceptual'):
        """
        Linear warmup - quadratic yerine basit linear
        """
        if epoch >= self.warmup_epochs:
            return 1.0
        
        # Basit linear artış
        multiplier = epoch / self.warmup_epochs
        
        return multiplier