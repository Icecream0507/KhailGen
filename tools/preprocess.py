import os
import librosa
import numpy as np
from tqdm import tqdm
import yaml

config_path = '/workspace/KhailGen-cloud/Khail/config.yaml'

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config(config_path)

def process_and_save_audio(audio_folder, processed_folder, sr, fixed_length):
    """
    加载、处理原始音频文件，将其分割为固定长度的片段，并保存为 .npy 格式。
    """
    os.makedirs(processed_folder, exist_ok=True)
    
    audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) 
                  if f.endswith('.wav') or f.endswith('.mp3')]

    print(f"Starting pre-processing of {len(audio_files)} audio files...")
    with tqdm(total=len(audio_files), desc="Processing audio files") as pbar:
        for file_path in audio_files:
            file_name = os.path.basename(file_path)
            base_name = os.path.splitext(file_name)[0]
            
            try:
                # 保持 mono=True，正确加载为单声道
                y, _ = librosa.load(file_path, sr=sr, mono=True)
                
                # 归一化到 [-1, 1]
                y = y / np.max(np.abs(y))
                
                # 计算需要分割的片段数量
                waveform_len = y.shape[0]
                num_segments = int(np.ceil(waveform_len / fixed_length))
                
                # 处理每个片段
                for i in range(num_segments):
                    start_idx = i * fixed_length
                    end_idx = start_idx + fixed_length
                    
                    # 提取片段
                    if end_idx <= waveform_len:
                        segment = y[start_idx:end_idx]
                    else:
                        # 最后一个片段，需要填充
                        segment = y[start_idx:]
                        pad_width = fixed_length - len(segment)
                        segment = np.pad(segment, (0, pad_width), mode='constant')
                    
                    # 生成唯一的文件名
                    segment_file_name = f"{base_name}_segment_{i:04d}.npy"
                    processed_file_path = os.path.join(processed_folder, segment_file_name)
                    
                    # 保存片段
                    np.save(processed_file_path, segment)
            
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                continue
            
            pbar.update(1)

    print("Audio pre-processing complete.")

if __name__ == '__main__':
    process_and_save_audio(
        audio_folder=config["audio_folder"],
        processed_folder=config["processed_folder"],
        sr=config["sample_rate"],
        fixed_length=config["waveform_length"]
    )