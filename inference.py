import torch
import argparse
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from train import create_model
from utils import psnr, bpp, save_image

def preprocess_image_rectangular(image_path: str, downsampling_rate: int):
    """
    Dikdörtgen bir görüntüyü, en-boy oranını koruyarak ve padding ekleyerek ön işler.
    
    Args:
        image_path (str): İşlenecek görüntünün yolu.
        downsampling_rate (int): Modelin toplam aşağı örnekleme oranı.

    Returns:
        torch.Tensor: Doldurulmuş, modelin beklediği formatta tensör.
        tuple: Orijinal (yükseklik, genişlik) boyutları.
    """
    to_tensor = transforms.ToTensor()
    image = Image.open(image_path).convert('RGB')
    
    # Görüntüyü tensöre çevir ve orijinal boyutları al
    tensor = to_tensor(image)
    original_h, original_w = tensor.shape[1], tensor.shape[2]
    
    # Gerekli yeni yüksekliği ve genişliği hesapla (downsampling_rate'in katı olmalı)
    pad_h = (downsampling_rate - original_h % downsampling_rate) % downsampling_rate
    pad_w = (downsampling_rate - original_w % downsampling_rate) % downsampling_rate
    
    # Padding uygula: (sol, sağ, üst, alt)
    # Görüntünün sağına ve altına padding ekliyoruz
    padding = (0, pad_w, 0, pad_h)
    padded_tensor = F.pad(tensor, padding, mode='constant', value=0)
    
    # Model bir batch beklediği için fazladan bir boyut ekle ve cihaza gönder
    return padded_tensor.unsqueeze(0), (original_h, original_w)

def main():
    parser = argparse.ArgumentParser(description="VQ-VAE Modeli ile Tek Görüntü Üzerinde Çıkarım (Dikdörtgen Destekli)")
    parser.add_argument('--model_path', type=str, required=True, help='Eğitilmiş modelin (.pth) yolu.')
    parser.add_argument('--image_path', type=str, required=True, help='Test edilecek görüntünün yolu.')
    parser.add_argument('--output_path', type=str, default='reconstructed_output.png', help='Yeniden oluşturulan görüntünün kaydedileceği yol.')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Model oluşturuluyor ve ağırlıklar yükleniyor...")
    model = create_model().to(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Görüntü işleniyor: {args.image_path}")
    downsampling_rate = model.encoder.downsampling_rate
    input_tensor, original_dims = preprocess_image_rectangular(args.image_path, downsampling_rate)
    input_tensor = input_tensor.to(device)
    
    print("Çıkarım yapılıyor...")
    with torch.no_grad():
        recon_tensor, indices, logits, _ = model(input_tensor)
        
        # Metrikleri hesaplamadan önce orijinal boyuta geri kırp
        original_h, original_w = original_dims
        # Yeniden yapılandırılmış görüntüden padding'i kırp
        recon_cropped = recon_tensor[:, :, :original_h, :original_w]
        # Orijinal tensörden padding'i kırp
        input_cropped = input_tensor[:, :, :original_h, :original_w]
        
        # Metrikleri kırpılmış görüntüler üzerinden hesapla
        psnr_value = psnr(input_cropped, recon_cropped)
        bpp_value = bpp(logits, indices, input_cropped.shape)
        compression_ratio = 24 / bpp_value if bpp_value > 0 else float('inf')

    print("\n--- Sonuçlar ---")
    print(f"Orijinal Boyut: {original_w}x{original_h}")
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"Bits Per Pixel (BPP): {bpp_value:.4f}")
    print(f"Teorik Sıkıştırma Oranı: ~{compression_ratio:.2f}x")
    print("----------------\n")
    
    save_image(recon_cropped, args.output_path)
    print(f"Yeniden oluşturulan görüntü (orijinal boyutta) şuraya kaydedildi: {args.output_path}")

if __name__ == "__main__":
    main()