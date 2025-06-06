import os
import argparse
import torch
import torch.optim as optim
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from utils.losses import CombinedLoss, AdaptiveLossWeights, WarmupLossScheduler
from models import VQCompressionModel
from models.encoder import Encoder
from models.decoder import Decoder
from models.vq import VectorQuantizerEMA
from models.entropy import EntropyModel
from utils import create_dataloaders, psnr, update_progressive_augmentation, calculate_entropy_loss
from config import MODEL_CONFIG, TRAIN_CONFIG, MODEL_SAVE_DIR, LOSS_CONFIG

def parse_args():
    parser = argparse.ArgumentParser(description='VQ-VAE Gelişmiş Görüntü Sıkıştırma Modeli Eğitimi')
    parser.add_argument('--train_dir', type=str, default='data/DIV2K/DIV2K_train_HR', help='Eğitim veri seti dizini')
    parser.add_argument('--val_dir', type=str, default='data/DIV2K/DIV2K_valid_HR', help='Doğrulama veri seti dizini')
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'], help='Batch boyutu')
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'], help='Eğitim epoch sayısı')
    parser.add_argument('--lr', type=float, default=TRAIN_CONFIG['learning_rate'], help='Öğrenme oranı')
    parser.add_argument('--save_dir', type=str, default=MODEL_SAVE_DIR, help='Model kayıt dizini')
    parser.add_argument('--resume', type=str, default=None, help='Eğitimi devam ettirmek için checkpoint dosyası')
    parser.add_argument('--no_cuda', action='store_true', help='CUDA kullanımını devre dışı bırak')
    parser.add_argument('--use_adaptive_loss', action='store_true', default=LOSS_CONFIG['adaptive']['enabled'], help='Adaptive loss weights kullan')
    parser.add_argument('--use_warmup', action='store_true', default=LOSS_CONFIG['warmup']['enabled'], help='Loss warmup scheduler kullan')
    
    return parser.parse_args()

def create_model():
    """Model bileşenlerini oluşturur ve birleştirir"""
    encoder = Encoder(
        in_channels=MODEL_CONFIG['encoder']['in_channels'],
        latent_channels=MODEL_CONFIG['encoder']['latent_channels'],
        num_layers=MODEL_CONFIG['encoder']['num_layers'],
        downsampling_rate=MODEL_CONFIG['encoder']['downsampling_rate'],
        use_attention=MODEL_CONFIG['encoder']['use_attention']
    )
    
    vq = VectorQuantizerEMA(
        num_embeddings=MODEL_CONFIG['vq']['num_embeddings'],
        embedding_dim=MODEL_CONFIG['vq']['embedding_dim'],
        commitment_coef=MODEL_CONFIG['vq']['commitment_coef'],
        debug=True
    )
    
    # Encoder'ın downsampling channel'larını al
    encoder_downsampling_channels = encoder.get_downsampling_channels()
    print(f"Encoder downsampling channels: {encoder_downsampling_channels}")
    
    decoder = Decoder(
        out_channels=MODEL_CONFIG['decoder']['out_channels'],
        latent_channels=MODEL_CONFIG['decoder']['latent_channels'],
        num_layers=MODEL_CONFIG['decoder']['num_layers'],
        post_vq_layers=MODEL_CONFIG['decoder']['post_vq_layers'],
        downsampling_rate=MODEL_CONFIG['decoder']['downsampling_rate'],
        use_attention=MODEL_CONFIG['decoder']['use_attention'],
        encoder_channels=encoder_downsampling_channels  # Doğru channel bilgisini geç
    )
    
    entropy_model = EntropyModel(
        num_embeddings=MODEL_CONFIG['entropy_model']['num_embeddings'],
        num_layers=MODEL_CONFIG['entropy_model']['num_layers'],
        hidden_dim=MODEL_CONFIG['entropy_model']['hidden_dim'],
        num_heads=MODEL_CONFIG['entropy_model']['num_heads']
    )
    
    model = VQCompressionModel(
        encoder=encoder,
        vq=vq,
        decoder=decoder,
        entropy_model=entropy_model
    )
    
    return model

def create_loss_function(device, use_adaptive=False): # Burası zaten false'ta adaptive weights kullanılmayacak
    """Loss fonksiyonu oluştur"""
    if use_adaptive:
        initial_weights = LOSS_CONFIG['weights']
        
        base_criterion = CombinedLoss(
            l1_weight=1.0,
            l2_weight=1.0,
            perceptual_weight=1.0,
            ms_ssim_weight=1.0,
            vq_weight=1.0,
            entropy_weight=1.0
        ).to(device)
        
        adaptive_weights = AdaptiveLossWeights(
            initial_weights, 
            adaptation_rate=LOSS_CONFIG['adaptive']['adaptation_rate']
        ).to(device)
        
        return base_criterion, adaptive_weights
    else:
        criterion = CombinedLoss(
            l1_weight=LOSS_CONFIG['weights']['l1_weight'],
            l2_weight=LOSS_CONFIG['weights']['l2_weight'],
            perceptual_weight=LOSS_CONFIG['weights']['perceptual_weight'],
            ms_ssim_weight=LOSS_CONFIG['weights']['ms_ssim_weight'],
            vq_weight=LOSS_CONFIG['weights']['vq_weight'],
            entropy_weight=LOSS_CONFIG['weights']['entropy_weight']
        ).to(device)
        
        return criterion, None

def create_scheduler(optimizer):
    """Learning rate scheduler oluştur"""
    scheduler_config = TRAIN_CONFIG.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'cosine_restart')
    
    if scheduler_type == 'cosine_restart':
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config.get('T_0', 40),
            T_mult=scheduler_config.get('T_mult', 2),
            eta_min=scheduler_config.get('eta_min', 5e-7)
        )
    elif scheduler_type == 'step':
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 50),
            gamma=scheduler_config.get('gamma', 0.5)
        )
    else:
        scheduler = None
    
    return scheduler

def train_epoch(model, train_loader, optimizer, criterion, epoch, device, 
                adaptive_weights=None, warmup_scheduler=None):
    """Bir epoch için modeli eğitir"""
    model.train()
    
    loss_tracker = {
        'total': 0.0,
        'l1': 0.0,
        'l2': 0.0,
        'perceptual': 0.0,
        'ms_ssim': 0.0,
        'vq': 0.0,
        'entropy': 0.0
    }
    
    num_batches = len(train_loader)
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch_idx, images in enumerate(pbar):
        images = images.to(device)
        optimizer.zero_grad()
        
        # İleri geçiş
        recon, indices, logits, vq_loss = model(images)

        # VQ ema health debug için
        if batch_idx % 100 == 0:
            health = model.vq.get_health_status()
            if not health['healthy']:
                print(f"VQ Issues at epoch {epoch}, batch {batch_idx}: {health['issues']}")
        
        # Entropi kaybını hesapla
        entropy_loss = calculate_entropy_loss(logits, indices)
        
        # Loss hesaplama
        if adaptive_weights is not None:
            # Individual loss'ları hesapla
            individual_losses = {
                'l1': F.l1_loss(recon, images),
                'l2': F.mse_loss(recon, images),
                'vq': vq_loss,
                'entropy': entropy_loss
            }
            
            # Perceptual ve MS-SSIM loss'ları CombinedLoss içinde hesaplanacak
            _, detailed_losses = criterion(recon, images, vq_loss, entropy_loss)
            
            # Tüm loss'ları birleştir
            all_losses = {**individual_losses, **detailed_losses}
            
            # Adaptive weights ile total loss hesapla
            total_loss, current_weights = adaptive_weights(all_losses)
            
        else:
            # Warmup scheduler kontrolü
            if warmup_scheduler is not None:
                # Warmup multiplier'larını al
                perc_mult = warmup_scheduler.get_loss_multiplier(epoch, 'perceptual')
                ssim_mult = warmup_scheduler.get_loss_multiplier(epoch, 'ms_ssim')
                
                # Geçici olarak criterion'ın weight'lerini güncelle
                original_perc_weight = criterion.perceptual_weight
                original_ssim_weight = criterion.ms_ssim_weight
                
                criterion.perceptual_weight *= perc_mult
                criterion.ms_ssim_weight *= ssim_mult
                
                # Loss hesapla
                total_loss, detailed_losses = criterion(recon, images, vq_loss, entropy_loss)
                
                # Weight'leri geri yükle
                criterion.perceptual_weight = original_perc_weight
                criterion.ms_ssim_weight = original_ssim_weight
            else:
                # Normal loss hesaplama
                total_loss, detailed_losses = criterion(recon, images, vq_loss, entropy_loss)
        
        # Gradient clipping kontrol
        if TRAIN_CONFIG.get('gradient_clipping', {}).get('enabled', False):
            total_loss.backward()

            vq_params = list(model.vq.parameters())
            if vq_params:
                torch.nn.utils.clip_grad_norm_(vq_params, max_norm=0.5)  # VQ için daha düşük norm

            other_params = [p for name, p in model.named_parameters() if not name.startswith('vq.')]
            torch.nn.utils.clip_grad_norm_(other_params, TRAIN_CONFIG['gradient_clipping']['max_norm'])

            optimizer.step()
        else:
            total_loss.backward()
            optimizer.step()
        
        # Loss tracking güncelle
        if adaptive_weights is not None:
            for key in loss_tracker.keys():
                if key in all_losses:
                    loss_tracker[key] += all_losses[key].item()
                elif key == 'total':
                    loss_tracker[key] += total_loss.item()
        else:
            loss_tracker['total'] += total_loss.item()
            for key, value in detailed_losses.items():
                if key in loss_tracker:
                    loss_tracker[key] += value.item()
        
        # İlerleme çubuğunu güncelle
        pbar.set_postfix({
            'total': f"{total_loss.item():.4f}",
            'l1': f"{detailed_losses.get('l1', torch.tensor(0)).item():.4f}",
            'perc': f"{detailed_losses.get('perceptual', torch.tensor(0)).item():.4f}"
        })
    
    # Ortalama loss'ları hesapla
    for key in loss_tracker:
        loss_tracker[key] /= num_batches
    
    return loss_tracker

def validate_epoch(model, val_loader, criterion, device, adaptive_weights=None):
    """Modeli doğrulama setinde değerlendirir"""
    model.eval()
    
    loss_tracker = {
        'total': 0.0,
        'l1': 0.0,
        'l2': 0.0,
        'perceptual': 0.0,
        'ms_ssim': 0.0,
        'vq': 0.0,
        'entropy': 0.0
    }
    
    total_psnr = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        
        for batch_idx, images in enumerate(pbar):
            images = images.to(device)
            
            # İleri geçiş
            recon, indices, logits, vq_loss = model(images)
            
            # Entropi kaybını hesapla
            entropy_loss = calculate_entropy_loss(logits, indices)
            
            # Loss hesaplama
            if adaptive_weights is not None:
                individual_losses = {
                    'l1': F.l1_loss(recon, images),
                    'l2': F.mse_loss(recon, images),
                    'vq': vq_loss,
                    'entropy': entropy_loss
                }
                
                _, detailed_losses = criterion(recon, images, vq_loss, entropy_loss)
                all_losses = {**individual_losses, **detailed_losses}
                
                total_loss, _ = adaptive_weights(all_losses)
            else:
                total_loss, detailed_losses = criterion(recon, images, vq_loss, entropy_loss)
            
            # PSNR hesapla
            batch_psnr = psnr(images, recon)
            total_psnr += batch_psnr
            
            # Loss tracking güncelle
            if adaptive_weights is not None:
                for key in loss_tracker.keys():
                    if key in all_losses:
                        loss_tracker[key] += all_losses[key].item()
                    elif key == 'total':
                        loss_tracker[key] += total_loss.item()
            else:
                loss_tracker['total'] += total_loss.item()
                for key, value in detailed_losses.items():
                    if key in loss_tracker:
                        loss_tracker[key] += value.item()
            
            # İlerleme çubuğunu güncelle
            pbar.set_postfix({
                'val_loss': f"{total_loss.item():.4f}",
                'val_psnr': f"{batch_psnr:.2f}",
            })
    
    # Ortalama değerleri hesapla
    for key in loss_tracker:
        loss_tracker[key] /= num_batches
    
    # PSNR ekleme
    loss_tracker['psnr'] = total_psnr / num_batches
    
    return loss_tracker

def save_checkpoint(model, optimizer, epoch, loss_dict, save_path, 
                   adaptive_weights=None, scheduler=None):
    """Model durumunu kaydet"""

    health = model.vq.get_health_status()
    if not health['healthy']:
        print(f"WARNING: Saving unhealthy VQ model at epoch {epoch}")
        print(f"Issues: {health['issues']}")
        # Corruption counter'ı sıfırla
        model.vq.reset_corruption_counter()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_dict': loss_dict,
        'vq_health': health,
        'config': {
            'model_config': MODEL_CONFIG,
            'train_config': TRAIN_CONFIG,
            'loss_config': LOSS_CONFIG
        }
    }
    
    if adaptive_weights is not None:
        checkpoint['adaptive_weights_state_dict'] = adaptive_weights.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, save_path)
    print(f"Model kaydedildi: {save_path}")

def load_checkpoint(model, optimizer, checkpoint_path, adaptive_weights=None, scheduler=None):
    """Kayıtlı model durumunu yükle"""
    import torch
    import numpy as np
    
    # Önce basit yöntemle deneyelim
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("Checkpoint başarıyla yüklendi (weights_only=False)")
    except Exception as e1:
        print(f"İlk yükleme hatası: {e1}")
        
        # Safe globals ile deneme
        try:
            # Gerekli numpy fonksiyonlarını safe globals'a ekle
            safe_globals_list = [
                'numpy._core.multiarray._reconstruct',
                'numpy.core.multiarray._reconstruct',
                'numpy.ndarray',
                'numpy.dtype',
                'collections.OrderedDict',
                'torch._utils._rebuild_tensor_v2'
            ]
            
            # PyTorch'un kendi safe globals fonksiyonu
            with torch.serialization.safe_globals(safe_globals_list):
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            print("Checkpoint yüklendi (safe globals ile)")
            
        except Exception as e2:
            print(f"Safe globals hatası: {e2}")
            
            # Eski PyTorch versiyonları için
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                print("Checkpoint yüklendi (eski yöntem)")
            except Exception as e3:
                print(f"Tüm yükleme yöntemleri başarısız: {e3}")
                raise RuntimeError(f"Checkpoint yüklenemedi. Dosya bozuk olabilir: {checkpoint_path}")
    
    # Model state dict yükleme
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state dict başarıyla yüklendi")
    except Exception as e:
        print(f"Model state dict yükleme hatası: {e}")
        # Strict=False ile deneme
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing_keys:
            print(f"Eksik anahtarlar: {missing_keys}")
        if unexpected_keys:
            print(f"Beklenmeyen anahtarlar: {unexpected_keys}")
    
    # Optimizer state dict yükleme
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state dict başarıyla yüklendi")
    except Exception as e:
        print(f"Optimizer state dict yükleme hatası (devam ediliyor): {e}")
    
    epoch = checkpoint.get('epoch', 0)
    loss_dict = checkpoint.get('loss_dict', {'total': checkpoint.get('loss', 0)})
    
    # VQ health bilgisi kontrolü
    if 'vq_health' in checkpoint:
        vq_health = checkpoint['vq_health']
        print(f"VQ Health durumu yüklendi: {vq_health}")
    
    # Adaptive weights yükleme
    if adaptive_weights is not None and 'adaptive_weights_state_dict' in checkpoint:
        try:
            adaptive_weights.load_state_dict(checkpoint['adaptive_weights_state_dict'])
            print("Adaptive weights başarıyla yüklendi")
        except Exception as e:
            print(f"Adaptive weights yükleme hatası (devam ediliyor): {e}")
    
    # Scheduler yükleme
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler başarıyla yüklendi")
        except Exception as e:
            print(f"Scheduler yükleme hatası (devam ediliyor): {e}")
    
    print(f"Checkpoint yüklendi: epoch {epoch}, loss {loss_dict.get('total', 0):.6f}")
    return epoch, loss_dict

def save_sample_images(images, reconstructions, epoch, save_dir):
    """Örnek görüntüleri kaydet"""
    plt.figure(figsize=(12, 8))
    
    # Tensor'dan numpy'a dönüştür
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    reconstructions = reconstructions.cpu().permute(0, 2, 3, 1).numpy()
    
    num_images = min(4, images.shape[0])
    
    for i in range(num_images):
        # Orijinal görüntü
        plt.subplot(2, num_images, i + 1)
        plt.imshow(np.clip(images[i], 0, 1))
        plt.title(f"Orijinal {i+1}")
        plt.axis('off')
        
        # Yeniden oluşturulan görüntü
        plt.subplot(2, num_images, i + num_images + 1)
        plt.imshow(np.clip(reconstructions[i], 0, 1))
        plt.title(f"Recons {i+1}")
        plt.axis('off')
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/recon_epoch_{epoch}.png")
    plt.close()

def debug_vq_usage(model, sample_batch, device):
    """VQ kullanımını debug et"""
    model.eval()
    with torch.no_grad():
        x = sample_batch.to(device)
        
        # Encoder çıktısını al
        z, skip_features = model.encoder(x)
        print(f"Encoder output - shape: {z.shape}, mean: {z.mean():.4f}, std: {z.std():.4f}")
        print(f"Encoder output - min: {z.min():.4f}, max: {z.max():.4f}")
        
        # Pre-VQ layer
        z_pre = model.pre_vq(z)
        print(f"Pre-VQ output - shape: {z_pre.shape}, mean: {z_pre.mean():.4f}, std: {z_pre.std():.4f}")
        print(f"Pre-VQ output - min: {z_pre.min():.4f}, max: {z_pre.max():.4f}")
        
        # VQ embeddings kontrol
        embeddings = model.vq.embed.weight
        print(f"VQ Embeddings - shape: {embeddings.shape}, mean: {embeddings.mean():.4f}, std: {embeddings.std():.4f}")
        print(f"VQ Embeddings - min: {embeddings.min():.4f}, max: {embeddings.max():.4f}")
        
        # VQ işlemi
        z_q, indices, vq_loss = model.vq(z_pre)
        print(f"VQ indices - unique values: {torch.unique(indices).numel()}/{model.vq.num_embeddings}")
        print(f"VQ indices - most common: {torch.bincount(indices.flatten()).argmax().item()}")
        print(f"VQ loss: {vq_loss.item():.6f}")
        
        # Distance debugging
        flat_inputs = z_pre.permute(0, 2, 3, 1).contiguous().view(-1, z_pre.size(1))
        distances = torch.cdist(flat_inputs, embeddings)
        min_distances = distances.min(dim=1)[0]
        print(f"Min distances - mean: {min_distances.mean():.4f}, max: {min_distances.max():.4f}")

def main():
    args = parse_args()
    
    # CUDA kontrolü
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Cihaz: {device}")
    
    # Veri yükleyicileri oluştur
    train_loader, val_loader, train_dataset = create_dataloaders(
        args.train_dir,
        args.val_dir,
        batch_size=args.batch_size,
        max_train=TRAIN_CONFIG['max_train_samples'],
        max_val=TRAIN_CONFIG['max_val_samples'],
        current_epoch=0,
        use_progressive=True
    )
    
    print(f"Eğitim örnekleri: {len(train_loader.dataset)}")
    print(f"Doğrulama örnekleri: {len(val_loader.dataset)}")
    
    # Model oluştur
    model = create_model().to(device)
    print(f"Model parametreleri: {sum(p.numel() for p in model.parameters())}")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = create_scheduler(optimizer)
    
    # Loss fonksiyonu oluştur
    criterion, adaptive_weights = create_loss_function(device, args.use_adaptive_loss)
    
    # Warmup scheduler
    warmup_scheduler = None
    if args.use_warmup:
        warmup_scheduler = WarmupLossScheduler(
            warmup_epochs=LOSS_CONFIG['warmup']['epochs']
        )
    
    # Devam etme kontrolü
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        start_epoch, loss_dict = load_checkpoint(
            model, optimizer, args.resume, 
            adaptive_weights, scheduler
        )
        best_loss = loss_dict.get('total', float('inf'))
        start_epoch += 1
        
        # Progressive augmentation için epoch güncelle
        if hasattr(train_dataset, 'update_epoch'):
            train_dataset.update_epoch(start_epoch)
    
    # Eğitim döngüsü
    print(f"{args.epochs} epoch için eğitim başlıyor...")
    print(f"Loss konfigürasyonu: Adaptive={args.use_adaptive_loss}, Warmup={args.use_warmup}")
    
    patience = TRAIN_CONFIG['early_stopping_patience']
    patience_counter = 0
    
    sample_batch = next(iter(train_loader))
    print("=== VQ DEBUG ===")
    debug_vq_usage(model, sample_batch, device)
    print("================")

    for epoch in range(start_epoch, args.epochs):
        # Progressive augmentation güncelleme
        update_progressive_augmentation(train_dataset, epoch)

        # Epoch başında VQ durumu kontrol et
        vq_stats = model.vq.get_codebook_usage()
        if epoch % 10 == 0:
            print(f"VQ Stats - Usage: {vq_stats['usage_ratio']:.2%}, "
                  f"Entropy: {vq_stats['normalized_entropy']:.3f}, "
                  f"Corruptions: {vq_stats['corruption_count']}")
        
        # Eğitim
        train_losses = train_epoch(
            model, train_loader, optimizer, criterion, epoch, device,
            adaptive_weights, warmup_scheduler
        )
        
        # Doğrulama
        val_losses = validate_epoch(
            model, val_loader, criterion, device, adaptive_weights
        )
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()
        
        # Adaptive weights güncelleme
        if adaptive_weights is not None and epoch > 0:
            target_ratios = LOSS_CONFIG['adaptive'].get('target_ratios', {})
            if target_ratios:
                adaptive_weights.update_weights(val_losses, target_ratios)
        
        # Detaylı logging
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Train - Total: {train_losses['total']:.4f}, L1: {train_losses['l1']:.4f}, "
              f"Perc: {train_losses['perceptual']:.4f}, VQ: {train_losses['vq']:.4f}")
        print(f"Val   - Total: {val_losses['total']:.4f}, L1: {val_losses['l1']:.4f}, "
              f"Perc: {val_losses['perceptual']:.4f}, PSNR: {val_losses['psnr']:.2f} dB")
        
        # Örnek görüntüler kaydet
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                val_images = next(iter(val_loader)).to(device)
                val_recons, _, _, _ = model(val_images)
                save_sample_images(val_images, val_recons, epoch+1, 
                                 os.path.join(args.save_dir, 'images'))
        
        # Model kaydetme
        current_val_loss = val_losses['total']
        if current_val_loss < best_loss:
            best_loss = current_val_loss
            save_checkpoint(
                model, optimizer, epoch, val_losses,
                os.path.join(args.save_dir, 'best_model.pth'),
                adaptive_weights, scheduler
            )
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Belirli aralıklarla kaydetme
        if (epoch + 1) % TRAIN_CONFIG['save_every'] == 0:
            save_checkpoint(
                model, optimizer, epoch, val_losses,
                os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pth'),
                adaptive_weights, scheduler
            )
        
        # Early stopping kontrolü
        if patience_counter >= patience:
            print(f"Early stopping: {patience} epoch boyunca iyileşme yok.")
            break
    
    # Son modeli kaydet
    save_checkpoint(
        model, optimizer, args.epochs-1, val_losses,
        os.path.join(args.save_dir, 'final_model.pth'),
        adaptive_weights, scheduler
    )
    
    print("Eğitimi tamamlandı!")

if __name__ == "__main__":
    main()