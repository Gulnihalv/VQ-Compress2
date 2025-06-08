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
    
    def encode_with_model(self, indices: torch.Tensor, entropy_model, 
                         spatial_shape: Tuple[int, int]) -> bytes:
        """
        Entropy model kullanarak VQ indekslerini kodla
        
        Args:
            indices: VQ indices tensor [B, H*W]
            entropy_model: Trained entropy model
            spatial_shape: (H, W) spatial dimensions
        """
        if indices.numel() == 0:
            return b''
        
        batch_size = indices.shape[0]
        h, w = spatial_shape
        
        # Her sample için ayrı ayrı kodla
        encoded_batches = []
        
        for b in range(batch_size):
            sample_indices = indices[b].flatten()  # [H*W]
            
            # Entropy model'den olasılık dağılımlarını al
            with torch.no_grad():
                # Model'e batch dimension ekle
                model_input = indices[b:b+1]  # [1, H*W]
                probs_logits = entropy_model(model_input)  # [1, H*W, num_embeddings]
                probs_tensor = F.softmax(probs_logits[0], dim=-1)  # [H*W, num_embeddings]
            
            # Her pozisyon için ayrı kodlama
            encoded_bits = []
            low = 0
            high = self.max_value
            pending_bits = 0
            
            def output_bit(bit):
                nonlocal pending_bits
                encoded_bits.append(bit)
                for _ in range(pending_bits):
                    encoded_bits.append(1 - bit)
                pending_bits = 0
            
            # Her sembolü sırayla kodla
            for pos in range(len(sample_indices)):
                symbol = sample_indices[pos].item()
                
                # Bu pozisyon için olasılık dağılımı
                pos_probs = self._normalize_probabilities(probs_tensor[pos])
                
                # Sembolü kodla
                try:
                    low, high = self._encode_symbol_with_probs(symbol, pos_probs, low, high)
                except Exception as e:
                    logging.warning(f"Encoding error at position {pos}, symbol {symbol}: {e}")
                    # Fallback: uniform distribution
                    uniform_probs = np.ones(self.num_embeddings) / self.num_embeddings
                    low, high = self._encode_symbol_with_probs(symbol, uniform_probs, low, high)
                
                # Renormalization
                while True:
                    if high < self.half:
                        output_bit(0)
                    elif low >= self.half:
                        output_bit(1)
                        low -= self.half
                        high -= self.half
                    elif low >= self.quarter and high < self.three_quarter:
                        pending_bits += 1
                        low -= self.quarter
                        high -= self.quarter
                    else:
                        break
                    
                    low = 2 * low
                    high = 2 * high + 1
            
            # Son bitleri çıkar
            pending_bits += 1
            if low < self.quarter:
                output_bit(0)
            else:
                output_bit(1)
            
            # Bits to bytes
            encoded_bytes = self._bits_to_bytes(encoded_bits)
            encoded_batches.append({
                'data': encoded_bytes,
                'num_bits': len(encoded_bits),
                'num_symbols': len(sample_indices)
            })
        
        # Batch verilerini birleştir
        return self._pack_batch_data(encoded_batches, spatial_shape)
    

    def decode_with_model(self, encoded_data: bytes, entropy_model, 
                         batch_size: int, spatial_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Entropy model kullanarak kodlanmış veriyi çöz
        """
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
            
            current_indices = torch.zeros((1, h * w), dtype=torch.long)
            decoded_symbols = []
            
            low = 0
            high = self.max_value
            value = 0
            
            for i in range(min(self.precision_bits, len(encoded_bits))):
                value = (value << 1) | encoded_bits[i]
            
            bit_index = self.precision_bits
            
            pbar = tqdm(range(num_symbols), desc=f"Decoding Symbols (Batch {b+1}/{batch_size})", leave=False)
            
            for pos in pbar:
                
                current_indices[0, :len(decoded_symbols)] = torch.tensor(decoded_symbols, dtype=torch.long)
                
                with torch.no_grad():
                    try:
                        probs_logits = entropy_model(current_indices)
                        pos_probs_tensor = F.softmax(probs_logits[0, pos], dim=-1)
                        pos_probs = self._normalize_probabilities(pos_probs_tensor)
                    except:
                        pos_probs = np.ones(self.num_embeddings) / self.num_embeddings
                
                try:
                    symbol, low, high = self._decode_symbol_with_probs(
                        pos_probs, value, low, high)
                except Exception as e:
                    logging.warning(f"Decoding error at position {pos}: {e}")
                    symbol = 0
                
                decoded_symbols.append(symbol)
                
                while True:
                    if high < self.half: pass
                    elif low >= self.half:
                        value -= self.half
                        low -= self.half
                        high -= self.half
                    elif low >= self.quarter and high < self.three_quarter:
                        value -= self.quarter
                        low -= self.quarter
                        high -= self.quarter
                    else: break
                    
                    low = 2 * low
                    high = 2 * high + 1
                    value = 2 * value
                    
                    if bit_index < len(encoded_bits):
                        value |= encoded_bits[bit_index]
                        bit_index += 1
            
            decoded_tensor = torch.tensor(decoded_symbols[:h*w], dtype=torch.long)
            if len(decoded_tensor) < h * w:
                padding = torch.zeros(h * w - len(decoded_tensor), dtype=torch.long)
                decoded_tensor = torch.cat([decoded_tensor, padding])
            
            decoded_batches.append(decoded_tensor)
        
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