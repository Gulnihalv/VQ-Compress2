import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import struct
import json
import logging
from tqdm import tqdm

class ArithmeticCoder:

    def __init__(self, num_embeddings: int, precision_bits: int = 32, 
                 use_spatial_context: bool = True, context_size: int = 3):
        self.num_embeddings = num_embeddings
        self.precision_bits = precision_bits
        self.max_value = (1 << precision_bits) - 1
        self.quarter = 1 << (precision_bits - 2)
        self.half = 2 * self.quarter
        self.three_quarter = 3 * self.quarter
        
        # Spatial context parametreleri
        self.use_spatial_context = use_spatial_context
        self.context_size = context_size
        
        # Numerical stability için minimum olasılık
        self.min_prob = 1e-8
        
        logging.info(f"VQVAEArithmeticCoder initialized:")
        logging.info(f"  - Num embeddings: {num_embeddings}")
        logging.info(f"  - Precision bits: {precision_bits}")
        logging.info(f"  - Spatial context: {use_spatial_context}")
        logging.info(f"  - Context size: {context_size}")
    
    def _normalize_probabilities(self, probs: torch.Tensor) -> np.ndarray:
        """Olasılık dağılımını normalize et ve numpy'a çevir"""
        # GPU'dan CPU'ya ve numpy'a çevir
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        
        # Numerical stability
        probs = np.maximum(probs, self.min_prob)
        
        # Normalize
        probs = probs / np.sum(probs)
        
        # Son kontrol
        probs = np.maximum(probs, self.min_prob)
        probs = probs / np.sum(probs)
        
        return probs
    
    def _get_cumulative_probs(self, probs: np.ndarray) -> np.ndarray:
        """Kümülatif olasılık dağılımını hesapla"""
        # İlk eleman 0 olacak şekilde kümülatif hesapla
        cumulative = np.zeros(len(probs) + 1, dtype=np.float64)
        cumulative[1:] = np.cumsum(probs)
        
        # Son eleman tam 1.0 olmalı
        cumulative[-1] = 1.0
        
        return cumulative
    
    def _encode_symbol_with_probs(self, symbol: int, probs: np.ndarray, 
                                 low: int, high: int) -> Tuple[int, int]:
        """Verilen olasılık dağılımıyla sembolü kodla"""
        if not (0 <= symbol < len(probs)):
            raise ValueError(f"Invalid symbol {symbol}, must be in [0, {len(probs)-1}]")
        
        # Kümülatif olasılıkları hesapla
        cumulative = self._get_cumulative_probs(probs)
        
        # Aralık hesaplama
        range_size = high - low + 1
        
        # Yeni low ve high hesapla
        new_high = low + int(range_size * cumulative[symbol + 1]) - 1
        new_low = low + int(range_size * cumulative[symbol])
        
        # Aynı değer kontrolü (numerical precision)
        if new_low >= new_high:
            new_high = new_low + 1
            if new_high > self.max_value:
                new_high = self.max_value
                new_low = new_high - 1
        
        return new_low, new_high
    
    def _decode_symbol_with_probs(self, probs: np.ndarray, 
                                 value: int, low: int, high: int) -> Tuple[int, int, int]:
        """Verilen olasılık dağılımıyla sembolü çöz"""
        cumulative = self._get_cumulative_probs(probs)
        
        # Scaled value hesapla
        range_size = high - low + 1
        scaled_value = (value - low) / range_size
        
        # Hangi sembol aralığında olduğunu bul
        symbol = -1
        for i in range(len(cumulative) - 1):
            if cumulative[i] <= scaled_value < cumulative[i + 1]:
                symbol = i
                break
        
        # Son sembol için özel kontrol
        if symbol == -1 and scaled_value >= cumulative[-2]:
            symbol = len(probs) - 1
        
        if symbol == -1:
            # Fallback: en yakın sembolü bul
            distances = np.abs(cumulative[:-1] - scaled_value)
            symbol = np.argmin(distances)
        
        # Yeni aralığı hesapla
        new_high = low + int(range_size * cumulative[symbol + 1]) - 1
        new_low = low + int(range_size * cumulative[symbol])
        
        # Aynı değer kontrolü
        if new_low >= new_high:
            new_high = new_low + 1
        
        return symbol, new_low, new_high

    # arithmetic_coding.py -> ArithmeticCoder sınıfı içindeki bu fonksiyonu güncelleyin

    def encode_with_model(self, indices: torch.Tensor, entropy_model, 
                         spatial_shape: Tuple[int, int], device=None) -> bytes:
        """
        Entropy model kullanarak VQ indekslerini, decode ile simetrik,
        auto-regressive bir şekilde kodlar.
        """
        if device is None:
            device = torch.device("cpu")

        if indices.numel() == 0:
            return b''
        
        batch_size = indices.shape[0]
        h, w = spatial_shape
        num_symbols = h * w
        
        encoded_batches = []
        
        for b in range(batch_size):
            sample_indices = indices[b].flatten()
            encoded_bits = []
            low, high, pending_bits = 0, self.max_value, 0
            
            def output_bit(bit):
                nonlocal pending_bits
                encoded_bits.append(bit)
                for _ in range(pending_bits):
                    encoded_bits.append(1 - bit)
                pending_bits = 0
            
            CONTEXT_WINDOW_SIZE = 1024
            
            pbar = tqdm(range(num_symbols), desc=f"Encoding Symbols (Batch {b+1}/{batch_size})", leave=False)

            for pos in pbar:
                symbol = sample_indices[pos].item()
                
                with torch.no_grad():
                    if pos == 0:
                        # İlk sembol için uniform olasılık
                        pos_probs = np.ones(self.num_embeddings, dtype=np.float64) / self.num_embeddings
                    else:
                        # Sonraki semboller için bağlam penceresi kullan
                        start_pos = max(0, pos - CONTEXT_WINDOW_SIZE)
                        context_tensor = sample_indices[start_pos:pos].unsqueeze(0).to(device)
                        
                        probs_logits = entropy_model(context_tensor)
                        pos_probs_tensor = F.softmax(probs_logits[0, -1], dim=-1)
                        pos_probs = self._normalize_probabilities(pos_probs_tensor)
                
                # Sembolü bu anlık olasılıklarla kodla
                low, high = self._encode_symbol_with_probs(symbol, pos_probs, low, high)
                
                # Renormalizasyon
                while True:
                    if high < self.half:
                        output_bit(0)
                    elif low >= self.half:
                        output_bit(1)
                        low -= self.half; high -= self.half
                    elif low >= self.quarter and high < self.three_quarter:
                        pending_bits += 1
                        low -= self.quarter; high -= self.quarter
                    else:
                        break
                    low <<= 1; high <<= 1; high |= 1
            
            # Son bitleri çıkar
            pending_bits += 1
            if low < self.quarter:
                output_bit(0)
            else:
                output_bit(1)

            encoded_bytes = self._bits_to_bytes(encoded_bits)
            encoded_batches.append({
                'data': encoded_bytes,
                'num_bits': len(encoded_bits),
                'num_symbols': len(sample_indices)
            })
        
        return self._pack_batch_data(encoded_batches, spatial_shape)

    def decode_with_model(self, encoded_data: bytes, entropy_model, 
                         batch_size: int, spatial_shape: Tuple[int, int], device=None) -> torch.Tensor:
        """
        Entropy model kullanarak kodlanmış veriyi çöz
        """
        if device is None:
            device = torch.device("cpu")

        if not encoded_data:
            h, w = spatial_shape
            return torch.zeros((batch_size, h * w), dtype=torch.long)
        
        # Batch verilerini ayrıştır
        batch_data, metadata = self._unpack_batch_data(encoded_data)
        h, w = spatial_shape
        
        decoded_batches = []
        
        for b in range(batch_size):
            if b >= len(batch_data):
                decoded_batches.append(torch.zeros(h * w, dtype=torch.long))
                continue
            
            sample_data = batch_data[b]
            encoded_bits = self._bytes_to_bits(sample_data['data'], sample_data['num_bits'])
            num_symbols = sample_data['num_symbols']
            
            if not encoded_bits:
                decoded_batches.append(torch.zeros(h * w, dtype=torch.long))
                continue

            CONTEXT_WINDOW_SIZE = 1024
            
            current_indices = torch.zeros(num_symbols, dtype=torch.long, device=device)
            
            low, high, value, bit_index = 0, self.max_value, 0, self.precision_bits
            for i in range(min(self.precision_bits, len(encoded_bits))):
                value = (value << 1) | encoded_bits[i]
            
            pbar = tqdm(range(num_symbols), desc=f"Decoding Symbols (Batch {b+1}/{batch_size})", leave=False)
            
            for pos in pbar:

                if pos == 0:
                    # İLK SEMBOL: Henüz bir bağlam yok.
                    # Bu nedenle, model çağırmak yerine uniform olasılık kullanıyoruz.
                    pos_probs = np.ones(self.num_embeddings, dtype=np.float64) / self.num_embeddings
                else:
                    # Sadece son `CONTEXT_WINDOW_SIZE` kadar sembolü al.
                    start_pos = max(0, pos - CONTEXT_WINDOW_SIZE)
                    context_tensor = current_indices[start_pos:pos].unsqueeze(0) # [1, degisken_uzunluk]

                    with torch.no_grad():
                        # entropy_model'e doğrudan `current_indices` verilir.
                        probs_logits = entropy_model(context_tensor)
                        pos_probs_tensor = F.softmax(probs_logits[0, -1], dim=-1)
                        pos_probs = self._normalize_probabilities(pos_probs_tensor)
                
                symbol, low, high = self._decode_symbol_with_probs(pos_probs, value, low, high)  
                current_indices[pos] = symbol
                
                while True:
                    if high < self.half: pass
                    elif low >= self.half:
                        value -= self.half; low -= self.half; high -= self.half
                    elif low >= self.quarter and high < self.three_quarter:
                        value -= self.quarter; low -= self.quarter; high -= self.quarter
                    else: break
                    low <<= 1; high <<= 1; high |= 1; value <<= 1
                    if bit_index < len(encoded_bits):
                        value |= encoded_bits[bit_index]
                        bit_index += 1
            
            decoded_batches.append(current_indices)
        
        result = torch.stack(decoded_batches, dim=0)
        return result
    
    def _bits_to_bytes(self, bits: List[int]) -> bytes:
        """Bit listesini byte array'e çevir"""
        encoded_bytes = bytearray()
        for i in range(0, len(bits), 8):
            byte_bits = bits[i:i+8]
            while len(byte_bits) < 8:
                byte_bits.append(0)
            
            byte_value = 0
            for j, bit in enumerate(byte_bits):
                byte_value |= (bit << (7 - j))
            encoded_bytes.append(byte_value)
        
        return bytes(encoded_bytes)
    
    def _bytes_to_bits(self, data: bytes, num_bits: int) -> List[int]:
        """Byte array'i bit listesine çevir"""
        bits = []
        for byte_val in data:
            for i in range(8):
                bits.append((byte_val >> (7 - i)) & 1)
        return bits[:num_bits]
    
    def _pack_batch_data(self, batch_data: List[dict], spatial_shape: Tuple[int, int]) -> bytes:
        """Batch verilerini tek bir byte array'e paketle"""
        metadata = {
            'batch_size': len(batch_data),
            'spatial_shape': spatial_shape,
            'samples': []
        }
        
        packed_data = bytearray()
        
        for sample in batch_data:
            sample_meta = {
                'data_offset': len(packed_data),
                'data_length': len(sample['data']),
                'num_bits': sample['num_bits'],
                'num_symbols': sample['num_symbols']
            }
            metadata['samples'].append(sample_meta)
            packed_data.extend(sample['data'])
        
        # Metadata'yı JSON olarak encode et
        metadata_json = json.dumps(metadata).encode('utf-8')
        metadata_length = struct.pack('<I', len(metadata_json))
        
        return metadata_length + metadata_json + bytes(packed_data)
    
    def _unpack_batch_data(self, data: bytes) -> Tuple[List[dict], dict]:
        """Batch verilerini ayrıştır"""
        # Metadata'yı oku
        metadata_length = struct.unpack('<I', data[:4])[0]
        metadata_json = data[4:4+metadata_length]
        metadata = json.loads(metadata_json.decode('utf-8'))
        
        # Packed data
        packed_data = data[4+metadata_length:]
        
        # Her sample'ı ayrıştır
        batch_data = []
        for sample_meta in metadata['samples']:
            offset = sample_meta['data_offset']
            length = sample_meta['data_length']
            sample_data = {
                'data': packed_data[offset:offset+length],
                'num_bits': sample_meta['num_bits'],
                'num_symbols': sample_meta['num_symbols']
            }
            batch_data.append(sample_data)
        
        return batch_data, metadata