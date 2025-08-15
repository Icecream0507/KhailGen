from mutagen.mp4 import MP4
from mutagen.mp3 import MP3
from mutagen.id3 import USLT
import os
import shutil


# 探视M4A文件的元数据

def inspect_m4a_metadata(file_path):
    try:
        audio = MP4(file_path)
        
        # 打印所有可用的元数据标签
        print("=== 所有可用标签 ===")
        for tag in audio.tags:
            print(tag)  # 例如：'©nam', '©ART', '©lyr' 等

        # 打印常见元数据
        print("\n=== 详细元数据 ===")
        metadata_fields = {
            '©nam': '标题',
            '©ART': '艺术家',
            '©alb': '专辑',
            '©day': '年份',
            '©lyr': '歌词',
            '©gen': '流派',
            '©too': '编码工具',
            '©cmt': '注释',
            'trkn': '音轨号',
            'disk': '光盘号',
        }

        for tag, desc in metadata_fields.items():
            if tag in audio.tags:
                value = audio.tags[tag][0] if isinstance(audio.tags[tag], list) else audio.tags[tag]
                print(f"{desc}: {value}")

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")

def lrc_exists(file_path):
    """检查是否存在同名的歌词文件"""
    audio = MP4(file_path)
    return "©lyr" in audio.tags


def get_lyrics_from_mp3(file_path):
    try:
        audio = MP3(file_path)
        # ID3 标签中歌词的键通常是'USLT::eng'或'USLT'
        if 'USLT::eng' in audio.tags:
            # USLT 帧的内容是一个对象，其文本内容在 .text 属性中
            lyrics = audio.tags['USLT::eng'].text
            print(f"文件: {file_path}")
            print("歌词:")
            print(lyrics)
            return lyrics
        else:
            print(f"文件: {file_path} 中没有内嵌歌词。")
            return None
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None


def get_lyrics_from_m4a(file_path):
    try:
        audio = MP4(file_path)
        # 歌词通常存储在'©lyr'标签中
        if '©lyr' in audio:
            lyrics = audio['©lyr'][0]
            print(f"File: {file_path}")
            # print("Lyrics:")
            # print(lyrics)
            return lyrics
        else:
            print(f"File: {file_path} does not have embedded lyrics.")
            return None
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None
    

def store_lyrics_to_file(path):
    lrc_path = os.path.join(path, "lyrics")
    os.makedirs(lrc_path, exist_ok=True)  # 确保歌词目录存在
    for root, dirs, songs in os.walk(path):
        for song in songs:
            file_path = os.path.join(root, song)
            lyric = get_lyrics_from_m4a(file_path)
            save_path = os.path.join(lrc_path, song.replace(".m4a", ".txt"))
            if lyric:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(lyric)
                print(f"歌词已保存到 {save_path}.txt")
            else:
                print(f"未找到 {file_path} 的歌词或无法保存。")

def clean_lyrics(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"已删除歌词文件: {file_path}")
                except Exception as e:
                    print(f"删除 {file_path} 时出错: {e}")

# 使用示例
# file_name = "好不容易.m4a"
# get_lyrics_from_m4a(file_name)

# 使用示例
#file_name = r"data\01. 才二十三.m4a"
#inspect_m4a_metadata(file_name)
#get_lyrics_from_mp3(file_name)

#clean_lyrics(r".\data")  # 清理已有的歌词文件

store_lyrics_to_file(r".\data")  # 假设所有文件都在 data 目录下