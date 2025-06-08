import torch
import tarfile
import io
import json
import argparse
import zipfile
from compression.arithmetic_coding import ArithmeticCoder
from train import create_model
from inference import preprocess_image_rectangular

def compress_to_archive(indices_data: bytes, 
                        skip_features: dict, 
                        original_dims: tuple, 
                        output_path: str):
    """Tüm veriyi tek bir .tar.gz arşivine sıkıştırarak kaydeder."""
    with tarfile.open(output_path, "w:gz") as tar:
        # 1. Meta veriyi (orijinal boyutlar) JSON olarak kaydet
        metadata = {'original_dims': original_dims}
        metadata_bytes = json.dumps(metadata).encode('utf-8')
        metadata_stream = io.BytesIO(metadata_bytes)
        tarinfo = tarfile.TarInfo(name="metadata.json")
        tarinfo.size = len(metadata_bytes)
        tar.addfile(tarinfo, metadata_stream)

        # 2. Sıkıştırılmış indeksleri (bitstream) kaydet
        indices_stream = io.BytesIO(indices_data)
        tarinfo = tarfile.TarInfo(name="indices.bin")
        tarinfo.size = len(indices_data)
        tar.addfile(tarinfo, indices_stream)

        # 3. Skip features'ı manuel olarak zip'le
        # Önce tensörleri normal şekilde bellekteki bir dosyaya kaydet
        tensor_buffer = io.BytesIO()
        torch.save(skip_features, tensor_buffer)
        tensor_buffer.seek(0) # Buffer'ı başa sar
        
        # Şimdi bu bellek dosyasını bir zip arşivine yaz
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            # zip arşivi içine 'skips.pt' adında bir dosya olarak yaz
            zf.writestr('skips.pt', tensor_buffer.read())
        zip_buffer.seek(0)

        # Son olarak zip'lenmiş bu buffer'ı ana tar arşivine ekle
        tarinfo = tarfile.TarInfo(name="skips.ptz") # .ptz = pytorch zipped
        tarinfo.size = zip_buffer.getbuffer().nbytes
        tar.addfile(tarinfo, zip_buffer)

def main():
    parser = argparse.ArgumentParser(description="Görüntüyü sıkıştırıp tek bir dosyaya kaydeder.")
    parser.add_argument('--model_path', type=str, required=True, help='Eğitilmiş modelin (.pth) yolu.')
    parser.add_argument('--image_path', type=str, required=True, help='Sıkıştırılacak görüntünün yolu.')
    parser.add_argument('--output_path', type=str, default='compressed.vqvae', help='Sıkıştırılmış çıktının dosya yolu.')
    args = parser.parse_args()

    device = torch.device("cpu") # Sıkıştırma için CPU yeterli
    
    # Modeli yükle
    model = create_model().to(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Görüntüyü ön işle
    input_tensor, original_dims = preprocess_image_rectangular(args.image_path, model.encoder.downsampling_rate)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        # Encoder'ı çalıştırarak indeksleri ve skip feature'ları al
        z, skip_features = model.encoder(input_tensor)
        z_pre = model.pre_vq(z)
        _, indices, _ = model.vq(z_pre)

    # 1. Koderi başlat
    coder = ArithmeticCoder(num_embeddings=model.vq.num_embeddings)
    
    # 2. İndeksleri ve olasılık modelini kullanarak bitstream oluştur
    indices_bitstream = coder.encode_with_model(
        indices=indices,
        entropy_model=model.entropy_model,
        spatial_shape=(original_dims[0] // model.encoder.downsampling_rate, 
                       original_dims[1] // model.encoder.downsampling_rate)
    )

    skip_features_dict = {f'skip_{i}': s.cpu() for i, s in enumerate(skip_features)}
    
    # Tüm veriyi tek bir arşive paketle
    compress_to_archive(
        indices_data=indices_bitstream,
        skip_features=skip_features_dict,
        original_dims=original_dims,
        output_path=args.output_path
    )
    print(f"Görüntü başarıyla sıkıştırıldı ve şuraya kaydedildi: {args.output_path}")

if __name__ == "__main__":
    main()