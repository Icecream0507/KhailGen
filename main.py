import os
from src.train_utils_wave import train
# from src.train_utils import train

if __name__ == "__main__":
    # 配置文件的路径
    config_path = '/workspace/KhailGen-cloud/Khail/config.yaml'
    
    # 确保配置文件存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    # 启动训练
    train(config_path)