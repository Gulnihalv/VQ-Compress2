import torch
import tarfile
import json
import argparse
import zipfile
import io
from compression.arithmetic_coding import ArithmeticCoder
from train import create_model
from utils import save_image

def decompress_from_archive(archive_path: str) -> dict:
    """Arşivden tüm veriyi okur ve bir sözlük olarak döndürür."""
    extracted_data = {}
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is None: continue
            
            if member.name == "metadata.json":
                extracted_data['metadata'] = json.loads(f.read().decode('utf-8'))
            elif member.name == "indices.bin":
                # Veriyi byte olarak alıyoruz, decode etme işi sonraki adımda
                extracted_data['indices_data'] = f.read()
            elif member.name == "skips.ptz":
                # Ana tar dosyasından gelen zip buffer'ını oku
                zip_buffer = io.BytesIO(f.read())
                # Bu zip buffer'ını aç
                with zipfile.ZipFile(zip_buffer, 'r') as zf:
                    # İçindeki 'skips.pt' dosyasını oku
                    with zf.open('skips.pt') as inner_file:
                        # Son olarak tensörleri torch.load ile yükle
                        extracted_data['skip_features'] = torch.load(inner_file)
    
    return extracted_data

def main():
    parser = argparse.ArgumentParser(description="Sıkıştırılmış dosyadan görüntüyü yeniden oluşturur.")
    parser.add_argument('--model_path', type=str, required=True, help='Eğitilmiş modelin (.pth) yolu.')
    parser.add_argument('--input_path', type=str, required=True, help='Sıkıştırılmış dosyanın (.vqvae) yolu.')
    parser.add_argument('--output_path', type=str, default='reconstructed_final.png', help='Yeniden oluşturulan görüntünün dosya yolu.')
    args = parser.parse_args()

    device = torch.device("cpu")
    
    # Modeli yükle
    model = create_model().to(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Arşivden veriyi oku
    archive_data = decompress_from_archive(args.input_path)
    metadata = archive_data['metadata']
    indices_data = archive_data['indices_data']
    skip_features_dict = archive_data['skip_features']
    sorted_keys = sorted(skip_features_dict.keys(), key=lambda x: int(x.split('_')[1]))
    skip_features = [skip_features_dict[key].to(device) for key in sorted_keys]
    original_h, original_w = metadata['original_dims']

    # 1. Koderi başlat
    coder = ArithmeticCoder(num_embeddings=model.vq.num_embeddings)
    
    # 2. Bitstream'i decode ederek indeksleri geri al
    indices = coder.decode_with_model(
        encoded_data=indices_data,
        entropy_model=model.entropy_model,
        batch_size=1, # Tek bir görüntü işliyoruz
        spatial_shape=(original_h // model.encoder.downsampling_rate, 
                       original_w // model.encoder.downsampling_rate)
    ).to(device)
    
    with torch.no_grad():
        # İndekslerden ve skip feature'lardan görüntüyü yeniden oluştur
        downsampling_rate = model.encoder.downsampling_rate
        padded_h = (original_h + downsampling_rate - 1) // downsampling_rate
        padded_w = (original_w + downsampling_rate - 1) // downsampling_rate

        spatial_shape = (indices.shape[0], padded_h, padded_w)
        z_q = model.vq.lookup_indices(indices, spatial_shape)
        z_q = model.post_vq(z_q)
        
        recon_padded = model.decoder(z_q, skip_features)
        
        # Orijinal boyuta geri kırp
        recon_cropped = recon_padded[:, :, :original_h, :original_w]

    # Sonucu kaydet
    save_image(recon_cropped, args.output_path)
    print(f"Görüntü başarıyla yeniden oluşturuldu ve şuraya kaydedildi: {args.output_path}")

if __name__ == "__main__":
    main()