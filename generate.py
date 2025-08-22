import torch
import yaml
import os
import soundfile as sf  # 用于保存音频文件
from src.unet import SimpleUNet1D
from src.diffusion_model import DiffusionModel

# 定义一个函数来加载配置
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def generate_audio(config_path, num_samples=1):
    """
    加载训练好的模型并生成新的音频样本。
    
    Args:
        config_path (str): 配置文件的路径。
        num_samples (int): 要生成的音频样本数量。
    """
    # 1. 加载配置
    config = load_config(config_path)

    # 2. 准备设备和模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 实例化 UNet 和 DiffusionModel
    unet = SimpleUNet1D(
        in_channels=config["channels"],
        out_channels=config["channels"],
        time_embedding_dim=config["embedding_dim"],
        unet_channels=config["unet_channels"]
    ).to(device)
    
    # 确保模型文件存在
    model_path = config['model_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # 加载训练好的模型权重
    unet.load_state_dict(torch.load(model_path, map_location=device))
    unet.eval() # 切换到评估模式

    diffusion_model = DiffusionModel(unet, timesteps=config["timesteps"], device=device).to(device)

    # 3. 执行生成过程
    print(f"Starting audio generation...")
    # 定义要生成的波形形状
    sample_shape = (num_samples, config["channels"], config["waveform_length"])
    
    with torch.no_grad(): # 在生成过程中不需要计算梯度
        generated_waveform = diffusion_model.sample(sample_shape)
    
    # 将波形从 [channels, length] 转换为 [length, channels] 以便 soundfile 保存
    generated_waveform = generated_waveform.permute(0, 2, 1)

    for i in range(generated_waveform.shape[0]):
        # 将张量转换为 NumPy 数组
        audio_data = generated_waveform[i].cpu().numpy()
        
        # 定义输出文件名
        output_filename = f"generated_audio_{i+1}.wav"
        output_path = config["output_folder"] + os.path.sep + output_filename
        
        # 使用 soundfile 库保存音频
        sf.write(output_path, audio_data, config["sample_rate"])
        print(f"Saved generated audio to {output_path}")

if __name__ == "__main__":
    # 配置路径
    config_file_path = './config.yaml'
    
    generate_audio(config_file_path, num_samples=2) # 示例：生成2个音频样本