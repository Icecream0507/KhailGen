import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from src.data_utils import AudioDatasetWaveform
# 从 src.unet 替换为从 src.wavenet 导入 WaveNet1D 模型
from src.wavenet import WaveNet1D
from src.diffusion_model import DiffusionModel
import time
import yaml
import os
import soundfile as sf
from tensorboardX import SummaryWriter
import pynvml
import math

# =========================================================
# 以下函数与原始代码保持一致
# =========================================================

def get_gpu_memory_info():
    """获取指定GPU的内存使用情况"""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory_mib = info.free / (1024 ** 2)
        total_memory_mib = info.total / (1024 ** 2)
        
        print(f"GPU 0 - Total: {total_memory_mib:.2f} MiB, Free: {free_memory_mib:.2f} MiB")
        
        pynvml.nvmlShutdown()
        return free_memory_mib / 1024  # 返回可用内存，单位为 GiB
    
    except pynvml.NVMLError as error:
        print(f"Error getting GPU memory info: {error}")
        return 0

def wait_for_gpu_memory(min_free_gb, check_interval_seconds=20):
    """循环等待直到有足够的GPU内存"""
    required_memory_gb = min_free_gb
    
    print(f"\nChecking for required GPU memory...")
    while True:
        free_memory_gb = get_gpu_memory_info()
        
        if free_memory_gb >= required_memory_gb:
            print(f"Sufficient memory found: {free_memory_gb:.2f} GiB (Required: {required_memory_gb:.2f} GiB). Starting training.")
            break
        else:
            print(f"Insufficient memory. Waiting for {check_interval_seconds} seconds...")
            time.sleep(check_interval_seconds)
# =========================================================


def train(config_path):
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    writer = SummaryWriter(logdir=f'runs/audio_diffusion_exp_{time.strftime("%Y%m%d-%H%M%S")}')

    # 实例化数据集和模型
    print("Initializing dataset and model...")
    dataset = AudioDatasetWaveform(
        processed_folder=config["processed_folder"],
        fixed_length=config["waveform_length"]
    )
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"Using device: {device}")

    # === 在这里添加内存检测和等待 ===
    required_memory_gb = 8.0 
    wait_for_gpu_memory(required_memory_gb)
    # ====================================
    
    # 用 WaveNet1D 替换 SimpleUNet1D，并传入相应的配置参数
    wavenet = WaveNet1D(
        in_channels=config["channels"],
        out_channels=config["channels"],
        time_embedding_dim=config["embedding_dim"],
        res_channels=config["res_channels"],
        skip_channels=config["skip_channels"],
        num_res_layers=config["num_res_layers"],
        dilation_cycle=config["dilation_cycle"]
    ).to(device)
    
    # 将模型实例从 unet 更改为 wavenet
    diffusion_model = DiffusionModel(wavenet, timesteps=config["timesteps"], device=device).to(device)
    
    # 优化器使用 wavenet 的参数
    optimizer = torch.optim.Adam(wavenet.parameters(), lr=config["learning_rate"])
    loss_fn = nn.MSELoss()
    
    epochs = config["epochs"]
    
    scaler = GradScaler()

    # 创建保存音频的文件夹
    audio_save_dir = "/workspace/KhailGen-cloud/temp_noisy_audios"
    os.makedirs(audio_save_dir, exist_ok=True)
    
    # 定义需要保存的t值
    t_to_save = [50, 100, 200, 300, 400, 500]
    saved_t_vals = set()

    best_epoch_loss = float('inf')

    global_step = 0

    print("Starting training loop...")
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")

        total_epoch_loss = 0 # 新增：用于累加本周期内的损失

        # 使用enumerate(dataloader)来迭代，并使用try-except来处理内存错误
        data_iterator = enumerate(dataloader)
        while True:
            try:
                i, x_0 = next(data_iterator)

                x_0 = x_0.to(device)
                batch_size = x_0.shape[0]
                
                t = torch.randint(0, diffusion_model.timesteps, (batch_size,), device=device).long()
                
                with torch.amp.autocast('cuda'):
                    x_t, noise = diffusion_model.forward_process(x_0, t)
                    # 调用 wavenet 预测噪声
                    predicted_noise = wavenet(x_t, t)
                    loss = loss_fn(predicted_noise, noise)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # # --- 保存加噪音频的代码 ---
                # # 找到批次中t值为我们感兴趣的样本
                # for t_val in t_to_save:
                #    # 找到t等于t_val的样本在批次中的索引
                #    indices = (t == t_val).nonzero(as_tuple=True)[0]
                #    if indices.shape[0] > 0:
                #        # 仅保存第一个匹配的样本
                #        sample_idx = indices[0]
                #        sample_x_t = x_t[sample_idx].cpu().numpy().T # 转置以符合 soundfile 格式 (样本数, 通道数)
                #        file_path = os.path.join(audio_save_dir, f"t_{t_val}_epoch_{epoch}_batch_{i}.wav")
                #        sf.write(file_path, sample_x_t, config['sample_rate'])
                #        print(f"Saved noisy audio for t={t_val} to {file_path}")
                #        saved_t_vals.add(t_val)
                #    
                #    # 如果所有t_val都已保存，则跳出内层循环
                #    if len(saved_t_vals) == len(t_to_save):
                #        print("All required t_vals have been saved. Skipping further audio saves.")
                #        break
                #    # --- 结束 ---

                # 记录批次损失到 TensorBoard
                writer.add_scalar('Loss/Batch_Loss', loss.item(), global_step)
                total_epoch_loss += loss.item()
                global_step += 1

                if (i + 1) % 10 == 0:
                    print(f"Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

            except StopIteration:
                # 当迭代器耗尽时跳出循环
                break
            except RuntimeError as e:
                # 捕获CUDA out of memory错误
                if 'CUDA out of memory' in str(e):
                    print(f"{e} \n CUDA Out of Memory caught. Releasing cache and retrying...")
                    
                    # 关键步骤：清空CUDA内存缓存
                    torch.cuda.empty_cache()
                    
                    # 暂停一小段时间，给GPU时间释放内存
                    time.sleep(5)
                    
                    
                    # 继续下一次循环，会尝试处理同一个批次
                    continue
                else:
                    # 如果是其他RuntimeError，重新抛出
                    raise e

        avg_epoch_loss = total_epoch_loss / len(dataloader)
        writer.add_scalar('Loss/Epoch_Average_Loss', avg_epoch_loss, epoch)

        print(f"Epoch {epoch+1} finished, Average Loss: {avg_epoch_loss:.4f}")

        # 新增：实时保存最优模型
        if avg_epoch_loss < best_epoch_loss:
            best_epoch_loss = avg_epoch_loss
            # 保存 wavenet 的参数
            model_save_path = os.path.join(config['model_folder'], "wave_diffusion_best_model.pth")
            torch.save(wavenet.state_dict(), model_save_path)
            print(f"New best model saved! Average Loss: {best_epoch_loss:.4f}")

        model_save_path = os.path.join(config['model_folder'], "wave_diffusion_current_model.pth")
        torch.save(wavenet.state_dict(), model_save_path)

        
    print("\nTraining finished.")
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    version = 0
    while True:
        # 保存 wavenet 的参数
        model_save_path = f"diffusion_{time_stamp}_{config['waveform_length']/config['sample_rate']}_v{version}.pth"
        model_save_path = config['model_folder'] + "/" + model_save_path
        try:
            with open(model_save_path, 'x'):
                break
        except FileExistsError:
            version += 1

    torch.save(wavenet.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    writer.close()
