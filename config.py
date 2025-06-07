import torch

# Cihaz yapılandırması
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Veri yolu yapılandırması
TRAIN_DATA_DIR = "data/DIV2K/DIV2K_train_HR"
VAL_DATA_DIR = "data/DIV2K/DIV2K_valid_HR"
TEST_DATA_DIR = "data/test"
MODEL_SAVE_DIR = "checkpoints"

# Model yapılandırması
MODEL_CONFIG = {
    # Encoder parametreleri
    "encoder": {
        "in_channels": 3,
        "latent_channels": 192,  # 128'den artırıldı
        "num_layers": 4,
        "downsampling_rate": 8,
        "use_attention": True
    },
    
    # Vector Quantizer parametreleri
    "vq": {
        "num_embeddings": 2048,  # 1024'ten artırıldı
        "embedding_dim": 192,    # encoder ile uyumlu
        "commitment_coef": 0.25, # 0.1'den artırıldı
        "decay": 0.99           # EMA decay rate
    },
    
    # Decoder parametreleri
    "decoder": {
        "out_channels": 3,
        "latent_channels": 192,  # encoder ile uyumlu
        "num_layers": 4,
        "post_vq_layers": 2,    # VQ sonrası işleme katmanları
        "downsampling_rate": 8,
        "use_attention": True
    },
    
    # Entropi modeli parametreleri
    "entropy_model": {
        "num_embeddings": 2048,  # VQ ile uyumlu
        "num_layers": 4,         # 3'ten artırıldı
        "hidden_dim": 384,       # 256'dan artırıldı
        "num_heads": 8           # 4'ten artırıldı
    }
}

# GELİŞMİŞ LOSS KONFIGÜRASYONU
LOSS_CONFIG = {
    # Ana loss ağırlıkları
    "weights": {
        "l1_weight": 1.0,
        "l2_weight": 0.0,           # Bunu kapatıyorum. Gerektiğinde açılabilir.
        "perceptual_weight": 0.5,   # 0.2'den artırıldı. görsel kalite için
        "ms_ssim_weight": 0.0,     # Geçici olarak kapatıldı - perceptual ile çakışıyor 
        "vq_weight": 1.0,
        "entropy_weight": 0.01
    },
    
    # Perceptual loss konfigürasyonu
    "perceptual": {
        "layers": ['relu_2_1', 'relu_3_1'],     # Daha az katman - hız kazancı
        "layer_weights": [0.5, 0.5]  # Daha derin katmanlara daha fazla ağırlık. Ağırlıkları eşit yaptım ama 0.4 0.6 da yapılabilir.
    },
    
    # MS-SSIM konfigürasyonu
    "ms_ssim": {
        "data_range": 1.0,
        "weights": [0.3, 0.4, 0.3],  # 5 yerine 3 scale
        "enabled": False              # Şimdilik kapalı
    },
    
    # Adaptive loss konfigürasyonu
    "adaptive": {
        "enabled": False,             # Şimdilik kapalı - daha basit yaklaşım
        "adaptation_rate": 0.002,     # Daha yavaş adaptasyon
        "target_ratios": {
            "l1": 0.4,
            "perceptual": 0.4,
            "vq": 0.15,
            "entropy": 0.05
        }
    }, 
    
    # Warmup konfigürasyonu
    "warmup": {
        "enabled": True,
        "epochs": 10,                 # 15'ten 10'a düşürüldü
        "schedule": {
            "perceptual": "quadratic"  # Sadece perceptual için warmup
        }
    }
}

# Eğitim yapılandırması
TRAIN_CONFIG = {
    "batch_size": 24,          # 20'den arttırıldı
    "learning_rate": 8e-5,
    "weight_decay": 1e-5,
    "epochs": 150,
    "save_every": 5,
    "early_stopping_patience": 25,  # Daha uzun sabır
    
    # Geriye uyumluluk için (kullanılmıyor ama kalsın)
    "reconstruction_weight": 1.0,
    "vq_weight": 1.0,
    "entropy_weight": 0.01,
    "perceptual_weight": 0.5,     # Güncellendi
    "ms_ssim_weight": 0.0,        # Kapatıldı
    
    "max_train_samples": None,
    "max_val_samples": None,
    
    # Learning rate scheduler
    "scheduler": {
        "type": "cosine_restart",
        "T_0": 30,                 # Daha kısa cycle
        "T_mult": 2,
        "eta_min": 1e-6
    },
    
    # Gradient clipping - perceptual loss için kritik
    "gradient_clipping": {
        "enabled": True,
        "max_norm": 1.0
    }
}

# Veri ön işleme yapılandırması
DATA_CONFIG = {
    "image_size": 256,
    "crop_size": 256,
    "horizontal_flip": True,
    "augmentation_level": "progressive",
    
    # Progressive augmentation detayları - perceptual loss ile uyumlu
    "augmentation_schedule": {
        "epochs_0_24": { # İlk 24 epoch için daha hafif
            "rotation": 3,
            "brightness": 0.05,
            "contrast": 0.05,
            "saturation": 0.05,
            "hue": 0.02
        },
        "epochs_25_40": { # 25. epoktan 40'a kadar orta seviye
            "rotation": 8,
            "brightness": 0.1,
            "contrast": 0.1,
            "saturation": 0.1,
            "hue": 0.05,
            "gaussian_blur": 0.05
        },
        "epochs_41_120": { # 41'den sonrası daha agresif
            "rotation": 12,
            "brightness": 0.15,
            "contrast": 0.15,
            "saturation": 0.15,
            "hue": 0.08,
            "gaussian_blur": 0.2,
            "random_erasing": 0.05
        }
    }
}

TRAINING_STRATEGY = {
    "progressive_augmentation": {
        "enabled": True,
        "schedule": {
            "epochs_0_24": "custom",
            "epochs_25_40": "custom",
            "epochs_41_120": "custom",
        }
    }
}

# Test/değerlendirme yapılandırması
EVAL_CONFIG = {
    "batch_size": 12,  # Perceptual loss hesabı için daha düşük
    "metrics": ["psnr", "ms_ssim", "bpp"]
}

# Sıkıştırma yapılandırması
COMPRESSION_CONFIG = {
    "arithmetic_coder_precision": 32
}


DEMO_CONFIG = {
    'model_checkpoint_name': 'best_model.pth', 
    'sample_image_path': 'data/DIV2K/DIV2K_train_HR/0001.png', 
    'compressed_file_path': 'output_demo/compressed_image.vqvae',
    'decompressed_image_path': 'output_demo/reconstructed_image.png'
}