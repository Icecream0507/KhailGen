from pydub import AudioSegment
import os
import librosa

def m4a_to_wav(input_path):
    for root, dirs, files in os.walk(input_path):
        # 跳过已创建的wav目录（避免重复处理）
        if os.path.basename(root) == "wav":
            continue
            
        store_path = os.path.join(root, "wav")
        os.makedirs(store_path, exist_ok=True)
        
        for file in files:
            if file.lower().endswith('.m4a'):  # 只处理m4a文件
                
                song_path = os.path.join(root, file)
                output_path = os.path.join(store_path, file[:-4] + ".wav")
                
                # 检查输出文件是否已存在
                if os.path.exists(output_path):
                    print(f"已存在，跳过: {output_path}")
                    continue

                print(f"正在转换: {song_path} 到 {output_path}")
                    
                audio = AudioSegment.from_file(song_path, format="m4a")
                audio.export(
                    output_path, 
                    format="wav",
                    codec="pcm_s16le",
                    parameters=["-ar", "44100"]  # 明确采样率
                )
                print(f"✓ 转换成功: {file}")

def look_into_wav(input_path):
    y, sr = librosa.load(input_path, sr=None, mono=False)
    print(f"采样率: {sr}, 音频数据类型: {type(y)}, 音频数据形状: {y.shape}")
    print(f"音频数据前10个样本: {y[:10]}")
    

# 使用示例
input_folder = r".\data"
# m4a_to_wav(input_folder)

# look_into_wav(r"data\wav\01. 爱爱爱.wav")  # 替换为实际的wav文件路径