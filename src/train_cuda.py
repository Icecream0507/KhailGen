import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import os
from tqdm import tqdm
from torch.cuda.amp import GradScaler # 移除 autocast 的导入，直接使用 torch.amp.autocast
import yaml

config_path = '/workspace/KhailGen-cloud/Khail/config.yaml'

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config(config_path)


def pad_to_fixed_length_waveform(waveform, max_len):
    """
    将波形填充到固定长度 √
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
    """加载音频文件，转换为波形数据 √""" 
    # 假设文件已经是wav格式
    # 修改：设置mono=False以加载立体声
    procressed_dir = os.path.dirname(file_path) + "/processed"
    if not os.path.exists(procressed_dir):
        os.makedirs(procressed_dir)
    y, _ = librosa.load(file_path, sr=sr, mono=False)
    
    # 归一化到 [-1, 1]
    y = y / np.max(np.abs(y))

    save_path = os.path.join(procressed_dir, os.path.basename(file_path)).replace('.wav', '.npy')
    with open(save_path, 'wb') as f:
        np.save(f, y)

    return y

class AudioDatasetWaveform(Dataset):
    """
    自定义音频数据集，从预处理好的 .npy 文件中加载波形数据。
    """
    def __init__(self, processed_folder, fixed_length):
        self.processed_files = [os.path.join(processed_folder, f) for f in os.listdir(processed_folder) if f.endswith('.npy')]
        self.fixed_length = fixed_length

        print(f"Found {len(self.processed_files)} pre-processed audio files...")
        
        # 预加载所有波形数据到内存中以加速训练
        # 注意：如果数据集非常大，可以不预加载，而是在__getitem__中加载
        print("Loading all waveforms into memory...")
        self.waveforms = []
        with tqdm(total=len(self.processed_files), desc="Loading .npy files") as pbar:
            for file_path in self.processed_files:
                try:
                    waveform = np.load(file_path)
                    self.waveforms.append(waveform)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                pbar.update(1)
        
        # 验证所有波形的尺寸
        if not all(w.shape[1] == self.fixed_length for w in self.waveforms):
            print("Warning: Waveform lengths are not consistent. Some files might be corrupted.")
            
        print("All waveforms loaded.")

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        # 直接返回预加载的波形
        waveform = self.waveforms[idx]
        # 返回 (channels, length) 格式
        return torch.tensor(waveform, dtype=torch.float32)

# ==============================================================================
# 1. 模型核心：U-Net去噪网络（1D版本）
# ==============================================================================
class SimpleUNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim, unet_channels):
        super(SimpleUNet1D, self).__init__()

        # 时间步嵌入层
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(time_embedding_dim * 4, unet_channels[-1]) # 使用 unet_channels[-1] (256) 作为输出维度
        )

        # 编码器部分
        self.conv_in = self.conv_block(in_channels, unet_channels[1])
        self.down1 = self.conv_block(unet_channels[1], unet_channels[2])
        self.down2 = self.conv_block(unet_channels[2], unet_channels[3])

        # 解码器部分
        self.up1 = self.conv_block(unet_channels[3] + unet_channels[2], unet_channels[2])
        self.up2 = self.conv_block(unet_channels[2] + unet_channels[1], unet_channels[1])
        
        self.conv_out = nn.Conv1d(unet_channels[1], out_channels, kernel_size=3, padding=1)
        
    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x, t):
        # 将时间步t转换为位置编码，并嵌入
        time_embed = self.time_mlp(self.sinusoidal_embedding(t, self.time_mlp[0].in_features)).unsqueeze(-1)
        
        # 编码器
        d_in = self.conv_in(x) # size: 44100
        d1 = self.down1(F.max_pool1d(d_in, 2)) # size: 22050
        d2 = self.down2(F.max_pool1d(d1, 2)) # size: 11025
        
        # 将时间嵌入添加到中间层
        # 修复inplace操作错误：将 += 改为 = +
        d2 = d2 + time_embed
        
        # 解码器
        up1 = F.interpolate(d2, scale_factor=2, mode='linear', align_corners=True) # upsampled from d2, size: 22050
        up1 = self.up1(torch.cat([up1, d1], dim=1)) # 拼接跳跃连接d1, size: 22050
        
        up2 = F.interpolate(up1, scale_factor=2, mode='linear', align_corners=True) # upsampled from up1, size: 44100
        up2 = self.up2(torch.cat([up2, d_in], dim=1)) # 拼接跳跃连接d_in, size: 44100
        
        return self.conv_out(up2)

    def sinusoidal_embedding(self, t, dim):
        """
        生成时间步的位置编码
        """
        half_dim = dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=t.device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None].float() * embeddings[None, :] 
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
        
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
        noise = torch.randn_like(x_0).to(self.device)
        
        sqrt_alpha_prod_t = self.sqrt_alphas_cumprod[t].view(-1, self.unet.conv_out.out_channels, 1).to(self.device)
        sqrt_one_minus_alpha_prod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, self.unet.conv_out.out_channels, 1).to(self.device)
        
        # 公式：x_t = sqrt(alpha_t_bar) * x_0 + sqrt(1 - alpha_t_bar) * noise
        x_t = sqrt_alpha_prod_t * x_0 + sqrt_one_minus_alpha_prod_t * noise
        return x_t, noise
        
    def sample(self, shape):
        """从纯噪声开始，生成新的样本"""
        print("\nStarting sampling process to generate new audio...")
        x_t = torch.randn(shape).to(self.device)
        
        for t in tqdm(reversed(range(self.timesteps))):
            t_tensor = torch.tensor([t] * shape[0], device=self.device).long()
            
            predicted_noise = self.unet(x_t, t_tensor)
            
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            x_t = (1 / torch.sqrt(alpha)) * (x_t - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)
            
            if t > 0:
                noise = torch.randn_like(x_t).to(self.device)
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
        sr=config["sample_rate"],
        fixed_length=config["waveform_length"] # 使用配置的固定长度
    )
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # U-Net输入和输出通道是2 (双声道)
    unet = SimpleUNet1D(
        in_channels=config["channels"],
        out_channels=config["channels"],
        time_embedding_dim=config["embedding_dim"],
        unet_channels=config["unet_channels"]
    ).to(device)
    diffusion_model = DiffusionModel(unet, timesteps=config["timesteps"], device=device).to(device)
    
    optimizer = torch.optim.Adam(unet.parameters(), lr=config["learning_rate"])
    loss_fn = nn.MSELoss()
    
    epochs = config["epochs"]
    
    # 初始化 GradScaler 来处理混合精度训练的缩放
    scaler = GradScaler()

    print("Starting training loop...")
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        for i, x_0 in enumerate(dataloader):
            x_0 = x_0.to(device)
            batch_size = x_0.shape[0]
            
            t = torch.randint(0, diffusion_model.timesteps, (batch_size,), device=device).long()
            
            # 使用 autocast 开启混合精度训练
            with torch.amp.autocast('cuda'):
                x_t, noise = diffusion_model.forward_process(x_0, t)
                predicted_noise = unet(x_t, t)
                loss = loss_fn(predicted_noise, noise)
            
            # 使用 scaler 来缩放损失，进行反向传播和优化
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if (i + 1) % 10 == 0:
                print(f"Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch+1} finished, Final Loss: {loss.item():.4f}")

    print("\nTraining finished.")
    model_save_path = "diffusion_unet.pth"
    torch.save(unet.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train()
