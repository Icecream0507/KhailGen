import os
import shutil

def move_wav_to_folder(source_folder, target_folder):
    """
    将 source_folder 中的所有 .wav 文件移动到 target_folder 中。
    """
    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)

    print(f"正在将 .wav 文件从 {source_folder} 移动到 {target_folder}...")
    
    # 遍历源文件夹中的所有文件
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith('.wav'):
                print(f"找到文件: {file}")
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_folder, file)
                
                # 检查目标文件是否已存在
                if os.path.exists(target_path):
                    print(f"目标文件已存在，跳过: {target_path}")
                    continue
                
                # 移动文件
                shutil.move(source_path, target_path)
                print(f"已移动: {source_path} 到 {target_path}")


move_wav_to_folder(r"/workspace/KhailGen-cloud/data", r"/workspace/KhailGen-cloud/data/wav")