import os
import zipfile
import shutil
from tqdm import tqdm

# 脚本功能：解压 ZIP 文件，提取 M4A 文件到根目录，并清理解压目录


data_path = r"D:\Download\方大同"  # ZIP 文件所在目录
extract_to_path = r".\data"  # 解压后的数据目录（建议用绝对路径）
zip_password = b"9898"  # 解压密码（必须是 bytes 类型）

def unzip_all_zips(zip_search_path, extract_to_path):
    # 确保目标目录存在
    os.makedirs(extract_to_path, exist_ok=True)

    # 先统计所有 ZIP 文件数量（用于进度条）
    zip_files = []
    for root, dirs, files in os.walk(zip_search_path):
        for file in files:
            if file.endswith('.zip'):
                zip_files.append(os.path.join(root, file))

    # 使用 tqdm 创建进度条
    with tqdm(total=len(zip_files), desc="解压 ZIP 文件", unit="file") as pbar:
        for zip_file_path in zip_files:
            try:
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    # 尝试用密码解压
                    zip_ref.extractall(extract_to_path, pwd=zip_password)
                pbar.update(1)  # 更新进度条
                pbar.set_postfix(file=os.path.basename(zip_file_path))  # 显示当前文件名
            except zipfile.BadZipFile:
                print(f"\n错误: {zip_file_path} 不是有效的 ZIP 文件")
            except RuntimeError as e:
                if "password" in str(e).lower():
                    print(f"\n密码错误或未提供密码: {zip_file_path}")
                else:
                    print(f"\n解压失败: {zip_file_path} - {e}")
            except Exception as e:
                print(f"\n解压 {zip_file_path} 时出错: {e}")

def get_m4a_out_of_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.m4a'):
                try: 
                    file_path = os.path.join(root, file)
                    shutil.move(file_path, os.path.join(folder_path))  # 移动到根目录
                    print(f"已移动文件: {file_path} 到 {folder_path}")
                except Exception as e:
                    print(f"移动文件 {file_path} 时出错: {e}")

def clean(target_dir):
    # 遍历目录，删除所有子文件夹
    for item in os.listdir(target_dir):
        item_path = os.path.join(target_dir, item)
        if os.path.isdir(item_path):  # 只处理目录
            shutil.rmtree(item_path)  # 递归删除非空目录
            print(f"已删除目录: {item_path}")

# 调用函数
#unzip_all_zips(data_path, extract_to_path)

#get_m4a_out_of_folder(extract_to_path)

clean(extract_to_path)  # 清理解压目录