import torch
import torch.nn as nn
from tqdm import tqdm
from src.unet import SimpleUNet1D

class DiffusionModel(nn.Module):
    def __init__(self, unet: SimpleUNet1D, timesteps=1000, device='cpu'):
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
        """线性beta调度。"""
        return torch.linspace(1e-5, 0.02, timesteps)
        
    def forward_process(self, x_0, t):
        """
        前向加噪过程。
        x_0 形状: (batch_size, channels, waveform_length)
        t 形状: (batch_size)
        """
        noise = torch.randn_like(x_0)
        
        # 扩展维度以匹配 x_0 的形状，以便进行广播
        sqrt_alpha_prod_t = self.sqrt_alphas_cumprod[t].unsqueeze(-1).unsqueeze(-1)
        sqrt_one_minus_alpha_prod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1).unsqueeze(-1)
        
        x_t = sqrt_alpha_prod_t * x_0 + sqrt_one_minus_alpha_prod_t * noise
        return x_t, noise
        
    def sample(self, shape):
        """
        从纯噪声开始，生成新的样本。
        shape 应为 (batch_size, channels, waveform_length)
        """
        print("\nStarting sampling process to generate new audio...")
        x_t = torch.randn(shape, device=self.device)
        
        for t in tqdm(reversed(range(self.timesteps))):
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            
            # 确保 unet 的输入维度正确
            predicted_noise = self.unet(x_t, t_tensor)
            
            # 扩展维度以进行广播
            alpha = self.alphas[t].view(1, 1, 1)
            alpha_cumprod = self.alphas_cumprod[t].view(1, 1, 1)
            beta = self.betas[t].view(1, 1, 1)
            
            x_t = (1 / torch.sqrt(alpha)) * (x_t - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)
            
            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = x_t + torch.sqrt(beta) * noise
        
        print("Sampling finished.")
        return x_t