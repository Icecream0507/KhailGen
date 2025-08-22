import subprocess
import time
import schedule
from datetime import datetime
import os

def run_git_commands():
    """执行Git命令序列的函数"""
    try:
        print(f"开始执行Git命令 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 设置工作目录（根据你的实际情况修改）
        workspace_path = "/workspace/KhailGen-cloud/Khail"
        
        # 切换到工作目录
        os.chdir(workspace_path)
        print(f"工作目录: {workspace_path}")
        
        # 执行 git add 命令
        print("执行: git add .")
        add_result = subprocess.run(["git", "add", "."], capture_output=True, text=True, timeout=60)
        
        
        # 执行 git commit 命令
        commit_message = f"自动提交 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        print(f"执行: git commit -m '{commit_message}'")
        commit_result = subprocess.run(["git", "commit", "-m", commit_message], 
                                      capture_output=True, text=True, timeout=60)
        
        if commit_result.returncode != 0:
            # 如果没有更改需要提交，git commit会返回非零状态码
            if "nothing to commit" in commit_result.stdout:
                print("没有需要提交的更改")
                return
            else:
                print(f"git commit 错误: {commit_result.stderr}")
                return
        
        # 执行 git push 命令
        print("执行: git push")
        push_result = subprocess.run(["git", "push", "origin", "Khail"], capture_output=True, text=True, timeout=120)
        
        if push_result.returncode != 0:
            print(f"git push 错误: {push_result.stderr}")
        else:
            print("Git操作成功完成!")
            
    except subprocess.TimeoutExpired:
        print("命令执行超时")
    except Exception as e:
        print(f"执行命令时发生错误: {e}")
    finally:
        print("-" * 50)

def main():
    print("Git自动提交脚本已启动")
    print("每10分钟自动执行: git add → git commit → git push")
    print("按 Ctrl+C 退出")
    print("-" * 50)
    
    # 安排每10分钟执行一次Git命令
    schedule.every(10).minutes.do(run_git_commands)
    
    # 立即执行一次
    run_git_commands()
    
    # 循环执行计划任务
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            print("\n程序已退出")
            break

if __name__ == "__main__":
    main()