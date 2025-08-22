import os
from huggingface_hub import create_repo, upload_folder

# 替换为你的用户名和数据集名称
# 仓库名称通常是你的用户名加上斜杠，再加上数据集名称
repo_id = "Ice144/KhailGen" 

# 你的数据集文件夹路径
# 请确保这个文件夹包含了所有你想要上传的文件
folder_path = "/workspace/KhailGen-cloud/data/wav" 

def upload_dataset_to_hub():
    """
    此函数将本地数据集文件夹上传到 Hugging Face Hub。
    它首先创建一个新的仓库，然后将文件夹中的所有内容上传到该仓库。
    """
    try:
        # 1. 创建一个新的数据集仓库
        print(f"正在创建新的仓库：{repo_id}")
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
        print("仓库创建成功或已存在。")

        # 2. 将整个文件夹上传到仓库
        print(f"正在上传文件夹 '{folder_path}' 到仓库 '{repo_id}'...")
        upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Initial dataset upload"
        )
        print("上传完成！")
        print(f"你可以在以下链接查看：https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        print(f"上传过程中发生错误：{e}")

# 运行上传函数
if __name__ == "__main__":
    # 在运行此脚本前，请确保你的本地已存在 'my_dataset' 文件夹
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在。请创建并放置你的数据集文件。")
    else:
        upload_dataset_to_hub()
