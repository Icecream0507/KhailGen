import os
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

def pad_to_fixed_length_waveform(waveform, max_len):
    """
    将波形填充到固定长度。此版本适用于单声道（1D）波形。
    """
    waveform_len = waveform.shape[0]
    if waveform_len < max_len:
        pad_width = max_len - waveform_len
        # 修正：填充代码以适用于一维数组
        waveform = np.pad(waveform, (0, pad_width), mode='constant')
    elif waveform_len > max_len:
        waveform = waveform[:max_len]
    return waveform

class AudioDatasetWaveform(Dataset):
    """自定义音频数据集，从预处理好的 .npy 文件中加载波形数据。"""
    def __init__(self, processed_folder, fixed_length):
        self.processed_files = [
            os.path.join(processed_folder, f) 
            for f in os.listdir(processed_folder) 
            if f.endswith('.npy')
        ]
        self.fixed_length = fixed_length

        print(f"Found {len(self.processed_files)} pre-processed audio files...")
        
        # 预加载所有波形数据到内存中以加速训练
        print("Loading all waveforms into memory...")
        self.waveforms = []
        with tqdm(total=len(self.processed_files), desc="Loading .npy files") as pbar:
            for file_path in self.processed_files:
                try:
                    waveform = np.load(file_path)
                    
                    # 修正：确保波形是正确的1D格式
                    if waveform.ndim == 2 and waveform.shape[0] == 1:
                        waveform = waveform.squeeze(0)
                    elif waveform.ndim != 1:
                        print(f"Warning: Skipping {file_path} due to unexpected dimensions: {waveform.shape}")
                        continue
                    
                    # 再次进行长度检查，以防预处理时出错
                    waveform = pad_to_fixed_length_waveform(waveform, self.fixed_length)
                    
                    self.waveforms.append(waveform)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                pbar.update(1)
        
        # 验证所有波形的尺寸
        if not all(w.shape[0] == self.fixed_length for w in self.waveforms):
            print("Warning: Waveform lengths are not consistent. Some files might be corrupted.")
            
        print("All waveforms loaded.")

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        waveform = self.waveforms[idx]
        # 修正：将一维 numpy 数组转换为带通道维度（channels=1）的 2D Tensor
        # 结果 Tensor 的形状为 [1, fixed_length]
        return torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)