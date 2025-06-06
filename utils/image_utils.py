import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from config import DATA_CONFIG, DEVICE, TRAIN_CONFIG, TRAINING_STRATEGY
import matplotlib.pyplot as plt
import random

def load_image(path):
    img = Image.open(path).convert('RGB')
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(img).unsqueeze(0)
    return tensor.to(DEVICE)

def save_image(tensor, filename):
    """Tensor'ı görüntü olarak kaydet"""
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    plt.imsave(filename, np.clip(img, 0, 1))

class EnhancedColorJitter:
    """Gelişmiş renk dönüşümleri"""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            if random.random() < 0.5:
                factor = random.uniform(1-self.brightness, 1+self.brightness)
                img = TF.adjust_brightness(img, factor)
            
            if random.random() < 0.5:
                factor = random.uniform(1-self.contrast, 1+self.contrast)
                img = TF.adjust_contrast(img, factor)
            
            if random.random() < 0.5:
                factor = random.uniform(1-self.saturation, 1+self.saturation)
                img = TF.adjust_saturation(img, factor)
            
            if random.random() < 0.3:
                factor = random.uniform(-self.hue, self.hue)
                img = TF.adjust_hue(img, factor)
        
        return img

class GaussianNoise:
    """Gaussian gürültü ekleme (tensor üzerinde çalışır)"""
    def __init__(self, mean=0.0, std=0.02, p=0.3):
        self.mean = mean
        self.std = std
        self.p = p
    
    def __call__(self, tensor):
        if random.random() < self.p:
            noise = torch.randn(tensor.size()) * self.std + self.mean
            return torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor

class GaussianBlur:
    """Gaussian blur ekleme"""
    def __init__(self, kernel_size=3, sigma_range=(0.1, 2.0), p=0.3):
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(*self.sigma_range)
            return TF.gaussian_blur(img, self.kernel_size, sigma)
        return img

class RandomErasing:
    """Random erasing augmentation"""
    def __init__(self, p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio
    
    def __call__(self, img):
        if random.random() < self.p:
            return transforms.RandomErasing(
                p=1.0, scale=self.scale, ratio=self.ratio
            )(img)
        return img

class RandomRotation90:
    """90 derece katları halinde döndürme"""
    def __init__(self, p=0.3):
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            angle = random.choice([90, 180, 270])
            return TF.rotate(img, angle)
        return img

class ImageDataset(Dataset):
    """
    Config-based progressive augmentation destekli görüntü dataset'i
    """
    def __init__(self, image_folder, transform=None, max_size=None, augmentation_params=None):
        """
        Args:
            image_folder (str): Görüntülerin bulunduğu klasör yolu
            transform (callable, optional): Her örnek için uygulanacak dönüşüm
            max_size (int, optional): Maksimum görüntü sayısı sınırlaması
            augmentation_params (dict, optional): Config'den gelen augmentation parametreleri
        """
        self.image_folder = image_folder
        self.augmentation_params = augmentation_params or {}
        self.max_size = max_size
        
        # Desteklenen görüntü uzantıları
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Dosya listesini al
        self.image_paths = []
        for root, dirs, files in os.walk(image_folder):
            for f in files:
                if os.path.splitext(f.lower())[1] in valid_exts:
                    self.image_paths.append(os.path.join(root, f))
        
        # Gerekirse veri setini sınırla
        if max_size and max_size < len(self.image_paths):
            self.image_paths = random.sample(self.image_paths, max_size)
        
        # Transform'u ayarla
        if transform is None:
            self.transform = self._get_config_based_transform()
        else:
            self.transform = transform
        
        print(f"Dataset initialized with {len(self.image_paths)} images")
        if self.augmentation_params:
            print(f"Augmentation params: {self.augmentation_params}")
    
    def _get_config_based_transform(self):
        """Config'den gelen parametrelerle transform oluştur"""
        config = DATA_CONFIG
        params = self.augmentation_params
        
        if not params:
            # Parametreler yoksa basit transform
            return transforms.Compose([
                transforms.RandomResizedCrop(config['crop_size']),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
        
        # Config parametrelerine göre transform listesi oluştur
        transform_list = []
        
        # Temel crop ve flip
        transform_list.extend([
            transforms.RandomResizedCrop(
                config['crop_size'], 
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(p=0.5)
        ])
        
        # Dikey flip (orta ve ileri seviye için)
        if params.get('rotation', 0) > 5:
            transform_list.append(transforms.RandomVerticalFlip(p=0.2))
        
        # 90 derece döndürme (ileri seviye için)
        if params.get('rotation', 0) > 8:
            transform_list.append(RandomRotation90(p=0.2))
        
        # Perspective transformation (en ileri seviye için)
        if params.get('rotation', 0) > 10:
            transform_list.append(
                transforms.RandomPerspective(distortion_scale=0.1, p=0.2)
            )
        
        # Renk augmentasyonları
        if any(key in params for key in ['brightness', 'contrast', 'saturation', 'hue']):
            transform_list.append(
                EnhancedColorJitter(
                    brightness=params.get('brightness', 0.1),
                    contrast=params.get('contrast', 0.1),
                    saturation=params.get('saturation', 0.1),
                    hue=params.get('hue', 0.02),
                    p=0.6
                )
            )
        
        # Rotasyon
        if params.get('rotation', 0) > 0:
            transform_list.append(
                transforms.RandomRotation(degrees=params['rotation'])
            )
        
        # Tensor'a dönüştür
        transform_list.append(transforms.ToTensor())
        
        # Gaussian blur (tensor üzerinde çalışır)
        if params.get('gaussian_blur', 0) > 0:
            blur_p = min(params['gaussian_blur'], 0.5)  # Maksimum %50
            transform_list.append(
                GaussianNoise(std=params['gaussian_blur'] * 0.05, p=blur_p)
            )
        
        # Random erasing (en son, tensor üzerinde)
        if params.get('random_erasing', 0) > 0:
            erasing_p = min(params['random_erasing'], 0.3)  # Maksimum %30
            transform_list.append(
                RandomErasing(p=erasing_p, scale=(0.02, 0.08))
            )
        
        return transforms.Compose(transform_list)
    
    def update_augmentation_params(self, new_params):
        """Augmentation parametrelerini güncelle"""
        self.augmentation_params = new_params
        self.transform = self._get_config_based_transform()
        print(f"Augmentation updated: {new_params}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Bir görüntüyü yükle ve dönüştür
        """
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
                
            return img
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # Hata durumunda rastgele başka bir resim döndür
            return self.__getitem__(random.randint(0, len(self.image_paths) - 1))

class ProgressiveImageDataset(ImageDataset):
    """Config-based progressive augmentation destekli dataset"""
    
    def __init__(self, image_folder, max_size=None, current_epoch=0):
        self.current_epoch = current_epoch
        
        # Config'den augmentation parametrelerini al
        augmentation_params = self._get_augmentation_params(current_epoch)
        
        super().__init__(image_folder, transform=None, max_size=max_size, 
                        augmentation_params=augmentation_params)
    
    def _get_augmentation_params(self, epoch):
        """Config'den epoch'a göre augmentation parametrelerini al"""
        if not TRAINING_STRATEGY.get("progressive_augmentation", {}).get("enabled", False):
            # Progressive augmentation kapalıysa, sabit parametreler kullan
            return DATA_CONFIG.get("augmentation_schedule", {}).get("epochs_0_24", {})

        schedule = DATA_CONFIG["augmentation_schedule"]

        # Epoch aralıklarını dinamik olarak kontrol edelim
        # schedule keys'i 'epochs_X_Y' formatında olduğunu varsayıyoruz
        for key in sorted(schedule.keys(), key=lambda x: int(x.split('_')[1])):
            start_epoch = int(key.split('_')[1])
            end_epoch = int(key.split('_')[2])
            if start_epoch <= epoch <= end_epoch:
                return schedule.get(key, {})

        # Eğer hiçbir aralığa denk gelmezse (epoch çok büyükse), son aralığı döndür
        # Veya varsayılan bir boş dict döndürebilirsiniz
        # Not: schedule dict'i düzgün tanımlandığında bu satıra gelinmemeli
        last_key = sorted(schedule.keys(), key=lambda x: int(x.split('_')[1]))[-1]
        return schedule.get(last_key, {})
    
    def update_epoch(self, epoch):
        """Epoch güncellendiğinde augmentation parametrelerini güncelle"""
        self.current_epoch = epoch
        new_params = self._get_augmentation_params(epoch)
        
        # Parametreler değiştiyse güncelle
        if new_params != self.augmentation_params:
            self.update_augmentation_params(new_params)

def create_dataloaders(train_dir, val_dir, batch_size=None, num_workers=None,
                      train_transform=None, val_transform=None, max_train=None, max_val=None,
                      current_epoch=0, use_progressive=True):
    """
    Config-based progressive augmentation ile gelişmiş veri yükleyicileri oluşturur
    
    Args:
        train_dir (str): Eğitim görüntülerinin dizini
        val_dir (str): Doğrulama görüntülerinin dizini
        batch_size (int): Batch boyutu (None ise config'den al)
        num_workers (int): Yükleme için paralel işçi sayısı (None ise config'den al)
        train_transform (callable, optional): Eğitim verilerine uygulanacak dönüşüm
        val_transform (callable, optional): Doğrulama verilerine uygulanacak dönüşüm
        max_train (int, optional): Maksimum eğitim görüntüsü sayısı
        max_val (int, optional): Maksimum doğrulama görüntüsü sayısı
        current_epoch (int): Mevcut epoch (progressive augmentation için)
        use_progressive (bool): Progressive augmentation kullan
        
    Returns:
        tuple: (train_loader, val_loader, train_dataset)
    """
    
    # Config'den varsayılan değerleri al
    if batch_size is None:
        batch_size = TRAIN_CONFIG.get("batch_size", 32)
    if num_workers is None:
        num_workers = TRAIN_CONFIG.get("num_workers", 4)
    if max_train is None:
        max_train = TRAIN_CONFIG.get("max_train_samples")
    if max_val is None:
        max_val = TRAIN_CONFIG.get("max_val_samples")
    
    # MPS uyumluluğu kontrolü (Apple Silicon)
    is_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    actual_workers = 0 if is_mps else num_workers
    use_pin_memory = not is_mps
    
    # Training dataset
    if use_progressive and TRAINING_STRATEGY.get("progressive_augmentation", {}).get("enabled", False):
        train_dataset = ProgressiveImageDataset(train_dir, max_train, current_epoch)
    else:
        # Sabit augmentation - config'den ilk seviye parametrelerini kullan
        if train_transform is None:
            default_params = DATA_CONFIG.get("augmentation_schedule", {}).get("epochs_0_24", {})
            train_dataset = ImageDataset(train_dir, transform=None, max_size=max_train, 
                                       augmentation_params=default_params)
        else:
            train_dataset = ImageDataset(train_dir, train_transform, max_train)
    
    # Validation dataset (augmentation yok)
    if val_transform is None:
        val_transform = transforms.Compose([
            transforms.Resize(DATA_CONFIG.get('image_size', 256)),
            transforms.CenterCrop(DATA_CONFIG.get('crop_size', 256)),
            transforms.ToTensor()
        ])
    
    val_dataset = ImageDataset(val_dir, val_transform, max_val, augmentation_params={})
    
    # DataLoader'ları oluştur
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=actual_workers,
        pin_memory=use_pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=actual_workers,
        pin_memory=use_pin_memory,
        drop_last=False
    )
    
    print(f"Created dataloaders - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    print(f"MPS detected: {is_mps}, Using {actual_workers} workers")
    
    return train_loader, val_loader, train_dataset

def update_progressive_augmentation(train_dataset, epoch):
    """Progressive augmentation için epoch güncellemesi"""
    if hasattr(train_dataset, 'update_epoch'):
        train_dataset.update_epoch(epoch)