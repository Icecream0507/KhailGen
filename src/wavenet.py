import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义swish激活函数
def swish(x):
    return x * torch.sigmoid(x)

# 带权重归一化和Kaiming初始化的卷积层
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)

# 零初始化的1x1卷积层
class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        return self.conv(x)

# WaveNet残差块，包含扩张卷积和时间步嵌入
class ResidualBlock(nn.Module):
    def __init__(self, res_channels, skip_channels, dilation, time_embed_dim):
        super(ResidualBlock, self).__init__()
        self.res_channels = res_channels
        self.fc_t = nn.Linear(time_embed_dim, res_channels)
        self.dilated_conv = Conv(res_channels, 2 * res_channels, kernel_size=3, dilation=dilation)
        self.res_conv = Conv(res_channels, res_channels, kernel_size=1)
        self.skip_conv = Conv(res_channels, skip_channels, kernel_size=1)

    def forward(self, input_data):
        x, time_embed = input_data
        h = x

        # 结合时间步嵌入
        time_part = self.fc_t(time_embed).unsqueeze(-1)
        h = h + time_part
        
        # 扩张卷积
        h = self.dilated_conv(h)
        
        # 门控tanh非线性
        out = torch.tanh(h[:, :self.res_channels, :]) * torch.sigmoid(h[:, self.res_channels:, :])

        # 残差和跳跃连接
        res = self.res_conv(out)
        skip = self.skip_conv(out)
        
        return (x + res) * math.sqrt(0.5), skip

# WaveNet残差组
class ResidualGroup(nn.Module):
    def __init__(self, res_channels, skip_channels, num_res_layers, dilation_cycle, time_embed_dim):
        super(ResidualGroup, self).__init__()
        self.num_res_layers = num_res_layers
        self.residual_blocks = nn.ModuleList()
        for i in range(num_res_layers):
            self.residual_blocks.append(
                ResidualBlock(res_channels, skip_channels, 2 ** (i % dilation_cycle), time_embed_dim)
            )

    def forward(self, x, time_embed):
        skip_outputs = 0
        h = x
        for block in self.residual_blocks:
            h, skip_n = block((h, time_embed))
            skip_outputs += skip_n
        
        return skip_outputs * math.sqrt(1.0 / self.num_res_layers)

# 完整WaveNet模型
class WaveNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim, res_channels, skip_channels, num_res_layers, dilation_cycle):
        super(WaveNet1D, self).__init__()
        self.time_embedding_dim = time_embedding_dim

        # 时间步嵌入层
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(time_embedding_dim * 4, res_channels)
        )

        # 初始1x1卷积
        self.init_conv = nn.Sequential(
            Conv(in_channels, res_channels, kernel_size=1),
            nn.ReLU()
        )

        # WaveNet核心部分
        self.residual_group = ResidualGroup(
            res_channels=res_channels,
            skip_channels=skip_channels,
            num_res_layers=num_res_layers,
            dilation_cycle=dilation_cycle,
            time_embed_dim=res_channels # 这里使用 time_mlp 的输出维度
        )

        # 最终卷积层
        self.final_conv = nn.Sequential(
            Conv(skip_channels, skip_channels, kernel_size=1),
            nn.ReLU(),
            ZeroConv1d(skip_channels, out_channels)
        )

    def forward(self, x, t):
        # 计算正弦时间步嵌入并进行转换
        time_embed = self.sinusoidal_embedding(t, self.time_embedding_dim)
        time_embed = self.time_mlp(time_embed)

        # 前向传播
        x = self.init_conv(x)
        x = self.residual_group(x, time_embed)
        return self.final_conv(x)

    def sinusoidal_embedding(self, t, dim):
        half_dim = dim // 2
        embeddings = math.log(10000.0) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t.float()[:, None] * embeddings[None, :] 
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings