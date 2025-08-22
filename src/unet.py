import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim, unet_channels):
        super(SimpleUNet1D, self).__init__()

        # 时间步嵌入层
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(time_embedding_dim * 4, unet_channels[-1])
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
        time_embed = self.time_mlp(self.sinusoidal_embedding(t, self.time_mlp[0].in_features)).unsqueeze(-1)
        
        # 编码器
        d_in = self.conv_in(x)
        d1 = self.down1(F.max_pool1d(d_in, 2))
        d2 = self.down2(F.max_pool1d(d1, 2))
        
        d2 = d2 + time_embed
        
        # 解码器
        up1 = F.interpolate(d2, scale_factor=2, mode='linear', align_corners=True)
        up1 = self.up1(torch.cat([up1, d1], dim=1))
        
        up2 = F.interpolate(up1, scale_factor=2, mode='linear', align_corners=True)
        up2 = self.up2(torch.cat([up2, d_in], dim=1))
        
        return self.conv_out(up2)

    def sinusoidal_embedding(self, t, dim):
        half_dim = dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=t.device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None].float() * embeddings[None, :] 
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings