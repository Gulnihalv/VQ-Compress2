import os
import glob
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.datasets.utils import download_url, extract_archive

class DIV2KDataset(Dataset):
    """DIV2K veri seti için özel Dataset sınıfı"""
    
    def __init__(self, root_dir=None, split='train', scale=None, transform=None, download=False):
        """
        Args:
            root_dir (string): DIV2K görüntülerinin bulunduğu veya indirileceği ana klasör
            split (string): 'train' veya 'valid' - hangi veri setinin kullanılacağı
            scale (int, optional): Alt örnekleme faktörü (2, 3 veya 4) - None ise HR görüntüler kullanılır
            transform (callable, optional): Örneklere uygulanacak dönüşüm
            download (bool): True ise, internet üzerinden indirilir
        """
        self.root_dir = root_dir or os.path.join(os.getcwd(), 'data', 'DIV2K')
        self.split = split
        self.scale = scale
        self.transform = transform
        
        # DIV2K URL'leri (resmi web sitesinden)
        self.urls = {
            'train_hr': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
            'valid_hr': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip',
            'train_lr_x2': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip',
            'valid_lr_x2': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip',
            'train_lr_x3': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip',
            'valid_lr_x3': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X3.zip',
            'train_lr_x4': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip',
            'valid_lr_x4': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip',
        }
        
        # Eğer indirme isteniyorsa, uygun dosyaları indir
        if download:
            self._download_dataset()
        
        # Görüntü dosyalarını bul
        self._find_images()
        
        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"Klasörde görüntü bulunamadı: {os.path.join(self.root_dir, self._get_image_dir())}. "
                "download=True olarak ayarlayın veya klasörün doğru olduğundan emin olun."
            )
    
    def _get_image_dir(self):
        """Görüntülerin bulunduğu klasörü döndürür"""
        if self.scale is None:
            # Yüksek çözünürlüklü görüntüler
            return f"DIV2K_{self.split}_HR"
        else:
            # Düşük çözünürlüklü görüntüler
            return f"DIV2K_{self.split}_LR_bicubic/X{self.scale}"
    
    def _download_dataset(self):
        """DIV2K veri setini indir"""
        os.makedirs(self.root_dir, exist_ok=True)
        
        # İndirilecek dosyaları belirle
        files_to_download = []
        if self.scale is None:
            # Sadece HR görüntüleri indir
            files_to_download.append(f"{self.split}_hr")
        else:
            # Hem HR hem de belirli ölçek için LR görüntüleri indir
            files_to_download.extend([f"{self.split}_hr", f"{self.split}_lr_x{self.scale}"])
        
        # Dosyaları indir ve çıkart
        for file_key in files_to_download:
            url = self.urls[file_key]
            filename = os.path.basename(url)
            download_path = os.path.join(self.root_dir, filename)
            
            # Zaten indirilmiş mi kontrol et
            extract_dir = os.path.join(self.root_dir, os.path.splitext(filename)[0])
            if os.path.exists(extract_dir):
                print(f"{filename} zaten çıkartılmış.")
                continue
            
            print(f"{filename} indiriliyor...")
            download_url(url, self.root_dir, filename)
            
            print(f"{filename} çıkartılıyor...")
            extract_archive(download_path, self.root_dir)
            
            # İsterseniz, ZIP dosyasını silebilirsiniz:
            # os.remove(download_path)
    
    def _find_images(self):
        """Görüntü dosyalarını bulur ve sıralar"""
        image_dir = os.path.join(self.root_dir, self._get_image_dir())
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        
        # DIV2K'da bazı görüntüler .png, bazıları .jpg olabilir
        if not self.image_paths:
            self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

def create_div2k_dataloaders(batch_size=16, train_transform=None, val_transform=None, 
                             max_train=None, max_val=None, patch_size=256, scale=None):
    """
    DIV2K veri setini yükler ve dataloader'lar oluşturur
    
    Args:
        batch_size (int): Batch boyutu
        train_transform (callable, optional): Eğitim örneklerine uygulanacak özel dönüşüm
        val_transform (callable, optional): Doğrulama örneklerine uygulanacak özel dönüşüm
        max_train (int, optional): Maksimum eğitim örneği sayısı 
        max_val (int, optional): Maksimum doğrulama örneği sayısı
        patch_size (int): Görüntülerden alınacak patch boyutu
        scale (int, optional): Alt örnekleme faktörü (2, 3, 4 veya None)
    
    Returns:
        train_loader, val_loader: Veri yükleyicileri
    """
    # Eğer transform belirtilmediyse varsayılan dönüşümleri kullan
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomCrop(patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
    
    if val_transform is None:
        val_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    # DIV2K veri setlerini yükle
    print("DIV2K veri seti yükleniyor...")
    
    train_dataset = DIV2KDataset(
        split='train',
        scale=scale,
        transform=train_transform,
        download=True
    )
    
    val_dataset = DIV2KDataset(
        split='valid',
        scale=scale,
        transform=val_transform,
        download=True
    )
    
    # Eğer maksimum örnek sayısı belirtildiyse alt küme oluştur
    if max_train is not None and max_train < len(train_dataset):
        indices = torch.randperm(len(train_dataset))[:max_train]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    
    if max_val is not None and max_val < len(val_dataset):
        indices = torch.randperm(len(val_dataset))[:max_val]
        val_dataset = torch.utils.data.Subset(val_dataset, indices)
    
    # DataLoader'ları oluştur
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,#memory yetmediğinden 4'ten 0'a düşürdüm
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0, #memory yetmediğinden 4'ten 0'a düşürdüm
        pin_memory=True
    )
    
    print(f"Eğitim örnekleri: {len(train_dataset)}")
    print(f"Doğrulama örnekleri: {len(val_dataset)}")
    
    return train_loader, val_loader

class RandomPatchesDataset(Dataset):
    """DIV2K veri setinden rastgele kesitler alan dataset sınıfı"""
    
    def __init__(self, div2k_dataset, patch_size=256, num_patches_per_image=16):
        """
        Args:
            div2k_dataset: Temel DIV2K veri seti
            patch_size (int): Her bir kesit boyutu 
            num_patches_per_image (int): Her bir görüntüden kaç adet kesit alınacağı
        """
        self.div2k_dataset = div2k_dataset
        self.patch_size = patch_size
        self.num_patches_per_image = num_patches_per_image
        
        # Tüm dönüşümleri kaydederek orijinal Transform'u bul
        original_transforms = div2k_dataset.transform.transforms if hasattr(div2k_dataset, 'transform') else None
        
        # ToTensor dışındaki dönüşümleri kaldır (biz kırpma işlemini kendimiz yapacağız)
        self.to_tensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.div2k_dataset) * self.num_patches_per_image
    
    def __getitem__(self, idx):
        # Hangi görüntü ve hangi patch?
        img_idx = idx // self.num_patches_per_image
        
        # Orijinal görüntüyü yükle (dönüşüm uygulamadan)
        # Div2K'dan doğrudan görüntü yolunu alabilmek için:
        if hasattr(self.div2k_dataset, 'image_paths'):
            img_path = self.div2k_dataset.image_paths[img_idx]
            image = Image.open(img_path).convert('RGB')
        else:
            # Eğer image_paths yoksa (örneğin, bir Subset kullanılıyorsa)
            temp_transform = self.div2k_dataset.transform
            self.div2k_dataset.transform = None
            image = self.div2k_dataset[img_idx]
            self.div2k_dataset.transform = temp_transform
        
        # Görüntü boyutları
        w, h = image.size
        
        # Rastgele bir kesit al (yeterli boyutta olduğunu kontrol et)
        if w < self.patch_size or h < self.patch_size:
            # Görüntü çok küçükse, boyutlarını artır
            image = image.resize((max(w, self.patch_size), max(h, self.patch_size)), Image.BICUBIC)
            w, h = image.size
        
        # Rastgele bir konum seç
        left = np.random.randint(0, w - self.patch_size + 1)
        top = np.random.randint(0, h - self.patch_size + 1)
        
        # Kesiti al
        patch = image.crop((left, top, left + self.patch_size, top + self.patch_size))
        
        # Veri artırma
        if np.random.random() > 0.5:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        
        if np.random.random() > 0.5:
            patch = patch.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Rotasyon (isteğe bağlı)
        rot_options = [0, 90, 180, 270]
        if np.random.random() > 0.5:
            angle = np.random.choice(rot_options)
            if angle > 0:
                patch = patch.rotate(angle)
        
        # Tensor'a dönüştür
        patch = self.to_tensor(patch)
        
        return patch

def create_div2k_patch_dataloaders(batch_size=16, patch_size=256, patches_per_image=16,
                                   max_train=None, max_val=None, scale=None):
    """
    DIV2K veri setinden rastgele kesitler alan dataloader'lar oluşturur
    
    Args:
        batch_size (int): Batch boyutu
        patch_size (int): Her bir kesit boyutu
        patches_per_image (int): Her bir görüntüden kaç kesit alınacağı
        max_train (int, optional): Maksimum eğitim görüntü sayısı
        max_val (int, optional): Maksimum doğrulama görüntü sayısı
        scale (int, optional): Alt örnekleme faktörü (2, 3, 4 veya None)
    
    Returns:
        train_loader, val_loader: Veri yükleyicileri
    """
    # Temel dönüşüm sadece ToTensor içermeli
    base_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # DIV2K veri setlerini yükle
    print("DIV2K veri seti yükleniyor...")
    
    train_base_dataset = DIV2KDataset(
        split='train',
        scale=scale,
        transform=base_transform,
        download=True
    )
    
    val_base_dataset = DIV2KDataset(
        split='valid',
        scale=scale,
        transform=base_transform,
        download=True
    )
    
    # Eğer maksimum görüntü sayısı belirtildiyse alt küme oluştur
    if max_train is not None and max_train < len(train_base_dataset):
        indices = torch.randperm(len(train_base_dataset))[:max_train]
        train_base_dataset = torch.utils.data.Subset(train_base_dataset, indices)
    
    if max_val is not None and max_val < len(val_base_dataset):
        indices = torch.randperm(len(val_base_dataset))[:max_val]
        val_base_dataset = torch.utils.data.Subset(val_base_dataset, indices)
    
    # Rastgele kesit veri setlerini oluştur
    train_dataset = RandomPatchesDataset(
        train_base_dataset,
        patch_size=patch_size,
        num_patches_per_image=patches_per_image
    )
    
    val_dataset = RandomPatchesDataset(
        val_base_dataset,
        patch_size=patch_size,
        num_patches_per_image=4  # Değerlendirme için daha az kesit yeterli
    )
    
    # DataLoader'ları oluştur
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Eğitim görüntüleri: {len(train_base_dataset)}")
    print(f"Doğrulama görüntüleri: {len(val_base_dataset)}")
    print(f"Eğitim kesitleri: {len(train_dataset)}")
    print(f"Doğrulama kesitleri: {len(val_dataset)}")
    
    return train_loader, val_loader

# Tüm veri setleri için genel bir arayüz
def create_dataloaders(dataset_name="div2k", batch_size=32, patch_size=256, 
                       max_train=None, max_val=None, train_transform=None, val_transform=None):
    """
    Belirtilen veri setini yükler ve uygun dataloader'lar oluşturur
    
    Args:
        dataset_name (str): Kullanılacak veri seti ("div2k", "cifar10" veya "kodak")
        batch_size (int): Batch boyutu
        patch_size (int): Görüntü yamaları için boyut
        max_train (int, optional): Maksimum eğitim örneği sayısı
        max_val (int, optional): Maksimum doğrulama örneği sayısı
        train_transform (callable, optional): Eğitim örneklerine uygulanacak özel dönüşüm
        val_transform (callable, optional): Doğrulama örneklerine uygulanacak özel dönüşüm
        
    Returns:
        train_loader, val_loader: Veri yükleyicileri
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "div2k":
        # DIV2K için varsayılan dönüşümler
        if train_transform is None:
            train_transform = transforms.Compose([
                transforms.RandomCrop(patch_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor()
            ])
        
        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.ToTensor()
            ])
        
        return create_div2k_dataloaders(
            batch_size=batch_size, 
            train_transform=train_transform,
            val_transform=val_transform,
            max_train=max_train,
            max_val=max_val,
            patch_size=patch_size
        )
    
    elif dataset_name == "div2k_patches":
        # DIV2K patch'ler için
        return create_div2k_patch_dataloaders(
            batch_size=batch_size,
            patch_size=patch_size,
            patches_per_image=16,
            max_train=max_train,
            max_val=max_val
        )
    
    elif dataset_name == "cifar10":
        # CIFAR-10 için
        from torchvision.datasets import CIFAR10
        
        # CIFAR-10 için varsayılan dönüşümler
        if train_transform is None:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        
        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.ToTensor()
            ])
        
        # CIFAR-10 veri setini yükle
        print("CIFAR-10 veri seti yükleniyor...")
        
        train_cifar = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        val_cifar = CIFAR10(root='./data', train=False, download=True, transform=val_transform)
        
        # Sadece görüntüleri alan wrapper sınıfı
        class ImageOnlyDataset(Dataset):
            def __init__(self, dataset):
                self.dataset = dataset
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                image, _ = self.dataset[idx]  # Etiketi yoksay
                return image  # Sadece görüntüyü döndür
        
        train_dataset = ImageOnlyDataset(train_cifar)
        val_dataset = ImageOnlyDataset(val_cifar)
    
    elif dataset_name == "kodak":
        # Kodak için
        # Kodak Dataset import eder
        from torchvision.datasets.folder import ImageFolder
        import requests
        
        # Kodak için varsayılan dönüşümler
        if train_transform is None:
            train_transform = transforms.Compose([
                transforms.ToTensor()
            ])
        
        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.ToTensor()
            ])
        
        # Kodak veri setini yükle
        kodak_dir = os.path.join(os.getcwd(), 'data', 'kodak')
        os.makedirs(kodak_dir, exist_ok=True)
        
        # Kodak görüntülerini indir
        kodak_urls = [f"http://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png" for i in range(1, 25)]
        
        # Zaten indirilmiş mi kontrol et
        if len(glob.glob(os.path.join(kodak_dir, "*.png"))) < 24:
            print("Kodak veri seti indiriliyor...")
            for i, url in enumerate(kodak_urls):
                img_path = os.path.join(kodak_dir, f"kodim{i+1:02d}.png")
                if not os.path.exists(img_path):
                    try:
                        response = requests.get(url)
                        if response.status_code == 200:
                            with open(img_path, 'wb') as f:
                                f.write(response.content)
                        else:
                            print(f"Görüntü indirilemedi: {url}")
                    except Exception as e:
                        print(f"Hata: {e}")
        
        # Özel dataset sınıfı
        class KodakDataset(Dataset):
            def __init__(self, root_dir, transform=None):
                self.root_dir = root_dir
                self.transform = transform
                self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*.png")))
            
            def __len__(self):
                return len(self.image_paths)
            
            def __getitem__(self, idx):
                img_path = self.image_paths[idx]
                image = Image.open(img_path).convert('RGB')
                
                if self.transform:
                    image = self.transform(image)
                
                return image
        
        # Kodak veri setini yükle
        print("Kodak veri seti yükleniyor...")
        
        # Kodak test veri seti
        val_dataset = KodakDataset(kodak_dir, transform=val_transform)
        
        # UYARI: Kodak genellikle sadece test için kullanılır
        print("UYARI: Kodak sadece test için kullanılmalıdır. Eğitim için DIV2K veri setini kullanın.")
        train_dataset = val_dataset  # Sadece gösterim için!
    
    else:
        raise ValueError(f"Desteklenmeyen veri seti: {dataset_name}. 'div2k', 'div2k_patches', 'cifar10' veya 'kodak' kullanın.")
    
    # Eğer maksimum örnek sayısı belirtildiyse alt küme oluştur
    if max_train is not None and max_train < len(train_dataset):
        indices = torch.randperm(len(train_dataset))[:max_train]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    
    if max_val is not None and max_val < len(val_dataset):
        indices = torch.randperm(len(val_dataset))[:max_val]
        val_dataset = torch.utils.data.Subset(val_dataset, indices)
    
    # DataLoader'ları oluştur
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Eğitim örnekleri: {len(train_dataset)}")
    print(f"Doğrulama örnekleri: {len(val_dataset)}")
    
    return train_loader, val_loader

# train_loader, val_loader = create_dataloaders(
#     dataset_name="div2k",  # DIV2K veri setini kullanın
#     batch_size=16,         # Batch boyutu
#     patch_size=256         # Görüntü yamalarının boyutu
# )


