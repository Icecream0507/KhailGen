import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import os
from pydub import AudioSegment
from tqdm import tqdm

# ==============================================================================
# 0. 前期准备：参数配置、音频处理与数据加载
# ==============================================================================

# 可配置的参数字典，方便调整模型行为
# 这里的 sample_rate 和 waveform_length 就是你所说的“合理压缩”的参数
config = {
    # 数据集配置
    "audio_folder": ".\data\wav_test",
    "sample_rate": 22050,      # 采样率，降低可以减少数据点，实现压缩
    "waveform_length": 44100,  # 波形固定长度，过长会增加计算量，可以裁剪或填充
    "channels": 2,             # 新增：音频通道数，2表示立体声

    # 模型配置
    "timesteps": 1000,
    "unet_channels": [2, 64, 128, 256], # UNet的通道数
    "embedding_dim": 128,      # 时间步嵌入的维度

    # 训练配置
    "batch_size": 4,
    "epochs": 10,
    "learning_rate": 1e-4,
}


def pad_to_fixed_length_waveform(waveform, max_len):
    """
    将波形填充到固定长度
    """
    # 注意：波形现在是 (channels, length) 格式
    waveform_len = waveform.shape[1]
    if waveform_len < max_len:
        pad_width = max_len - waveform_len
        waveform = np.pad(waveform, ((0, 0), (0, pad_width)), mode='constant')
    elif waveform_len > max_len:
        waveform = waveform[:, :max_len]
    return waveform

def process_audio_waveform(file_path, sr):
    """加载音频文件，转换为波形数据"""
    # 假设文件已经是wav格式
    # 修改：设置mono=False以加载立体声
    y, _ = librosa.load(file_path, sr=sr, mono=False)
    
    # 归一化到 [-1, 1]
    y = y / np.max(np.abs(y))
    return y

class AudioDatasetWaveform(Dataset):
    """自定义音频数据集，直接处理波形"""
    def __init__(self, audio_folder, sr=22050):
        self.audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder)]
        self.sr = sr
        
        # 为了保证一个batch内的波形尺寸一致，需要找到最长的波形
        print("Pre-processing audio files to determine max length...")
        self.waveforms = [process_audio_waveform(f, sr) for f in self.audio_files]
        self.max_len = max(w.shape[1] for w in self.waveforms)
        print(f"Max audio length (samples): {self.max_len}")
        
    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        waveform = self.waveforms[idx]
        waveform = pad_to_fixed_length_waveform(waveform, self.max_len)
        # 修改：返回 (channels, length) 格式，不需要unsqueeze
        return torch.tensor(waveform, dtype=torch.float32)

# ==============================================================================
# 1. 模型核心：U-Net去噪网络（1D版本）
# ==============================================================================
class SimpleUNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super(SimpleUNet1D, self).__init__()

        # 时间步嵌入层
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(time_embedding_dim * 4, time_embedding_dim)
        )

        # 编码器部分
        # 修改：使用动态通道数
        self.conv_in = self.conv_block(in_channels, 64)
        self.down1 = self.conv_block(64, 128)
        self.down2 = self.conv_block(128, 256)

        # 解码器部分
        self.up1 = self.conv_block(256 + 128, 128)
        self.up2 = self.conv_block(128 + 64, 64)
        # 修改：使用动态通道数
        self.conv_out = nn.Conv1d(64, out_channels, kernel_size=3, padding=1)
        
    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x, t):
        # 将时间步t转换为位置编码，并嵌入
        t_embed = self.time_mlp(self.sinusoidal_embedding(t, self.time_mlp[0].in_features)).unsqueeze(-1)
        
        # 编码器
        d_in = self.conv_in(x)
        d1 = self.down1(d_in)
        d2 = self.down2(F.max_pool1d(d1, 2))
        
        # 将时间嵌入添加到中间层
        d2 += t_embed
        
        # 解码器
        up1 = F.interpolate(d2, scale_factor=2, mode='linear', align_corners=True)
        up1 = self.up1(torch.cat([up1, d1], dim=1)) # 拼接跳跃连接
        
        up2 = F.interpolate(up1, scale_factor=2, mode='linear', align_corners=True)
        up2 = self.up2(torch.cat([up2, d_in], dim=1)) # 拼接跳跃连接
        
        return self.conv_out(up2)

    def sinusoidal_embedding(self, t, dim):
        """
        生成时间步的位置编码
        """
        # --- 新增：使用正弦位置编码生成时间步的嵌入向量 ---
        half_dim = dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
        # ---------------------------------------------------
        

# ==============================================================================
# 2. 扩散模型的前向与反向过程
# ==============================================================================
class DiffusionModel(nn.Module):
    def __init__(self, unet, timesteps=1000, device='cpu'):
        super(DiffusionModel, self).__init__()
        self.unet = unet
        self.timesteps = timesteps
        self.device = device
        
        # 定义一个简单的线性beta-schedule
        self.betas = self.linear_schedule(timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        
        # 预计算一些常用的变量
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def linear_schedule(self, timesteps):
        """线性beta调度"""
        return torch.linspace(1e-4, 0.02, timesteps)
        
    def forward_process(self, x_0, t):
        """前向加噪过程"""
        noise = torch.randn_like(x_0)
        
        # 修改：根据U-Net的输出通道数调整维度
        sqrt_alpha_prod_t = self.sqrt_alphas_cumprod[t].view(-1, self.unet.conv_out.out_channels, 1)
        sqrt_one_minus_alpha_prod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, self.unet.conv_out.out_channels, 1)
        
        # 公式：x_t = sqrt(alpha_t_bar) * x_0 + sqrt(1 - alpha_t_bar) * noise
        x_t = sqrt_alpha_prod_t * x_0 + sqrt_one_minus_alpha_prod_t * noise
        return x_t, noise
        
    def sample(self, shape):
        """从纯噪声开始，生成新的样本"""
        print("\nStarting sampling process to generate new audio...")
        x_t = torch.randn(shape).to(self.device)
        
        for t in tqdm(reversed(range(self.timesteps))):
            t_tensor = torch.tensor([t] * shape[0], device=self.device).long()
            
            # 使用U-Net预测噪声
            predicted_noise = self.unet(x_t, t_tensor)
            
            # 简化版采样：一步去噪
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            # 修改：根据U-Net的输出通道数调整维度
            x_t = (1 / torch.sqrt(alpha)) * (x_t - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)
            
            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = x_t + torch.sqrt(beta) * noise
        
        print("Sampling finished.")
        return x_t

# ==============================================================================
# 3. 训练循环
# ==============================================================================
def train():
    # 实例化数据集和模型
    print("Initializing dataset and model...")
    dataset = AudioDatasetWaveform(
        audio_folder=config["audio_folder"],
        sr=config["sample_rate"]
    )
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # U-Net输入和输出通道是2 (双声道)
    unet = SimpleUNet1D(
        in_channels=config["channels"],
        out_channels=config["channels"],
        time_embedding_dim=config["embedding_dim"]
    ).to(device)
    diffusion_model = DiffusionModel(unet, timesteps=config["timesteps"], device=device).to(device)
    
    optimizer = torch.optim.Adam(unet.parameters(), lr=config["learning_rate"])
    loss_fn = nn.MSELoss()
    
    epochs = config["epochs"]
    
    print("Starting training loop...")
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        for i, x_0 in enumerate(dataloader):
            x_0 = x_0.to(device)
            batch_size = x_0.shape[0]
            
            # 随机选择一个时间步t
            t = torch.randint(0, diffusion_model.timesteps, (batch_size,), device=device).long()
            
            # 前向过程：加噪
            x_t, noise = diffusion_model.forward_process(x_0, t)
            
            # 反向过程：U-Net预测噪声
            predicted_noise = unet(x_t, t)
            
            # 计算损失
            loss = loss_fn(predicted_noise, noise)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 10 == 0:
                print(f"Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch+1} finished, Final Loss: {loss.item():.4f}")

    print("\nTraining finished.")
    # --- 新增：保存训练好的模型 ---
    model_save_path = "diffusion_unet.pth"
    torch.save(unet.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    # --------------------------------

    # 生成一个新样本
    # 注意：这里的shape需要根据你的数据集来确定
    # print("Generating a new audio sample...")
    # shape = (1, config["channels"], dataset.max_len)
    # generated_audio_waveform = diffusion_model.sample(shape)
    # # 将波形保存为音频文件
    # # librosa.output.write_wav("generated_audio.wav", generated_audio_waveform.squeeze().cpu().numpy(), config["sample_rate"])


# 在你的环境中运行
if __name__ == "__main__":
    train()
