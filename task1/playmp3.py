# 导入所需库：pygame处理音频，time控制播放状态检测
import pygame
import time

# ==================== 配置项（仅需修改这一行）====================
MP3_FILE_PATH = "lovesong.mp3"  # 替换为你的MP3文件路径（相对/绝对路径均可）
# 相对路径：MP3文件和代码文件在同一文件夹，直接写文件名（如上面）
# 绝对路径示例：MP3_FILE_PATH = "E:/音乐/lovesong.mp3" （注意用/或\\，避免\转义问题）
# ================================================================

def play_mp3_file(mp3_path):
    """
    核心函数：播放指定的MP3文件
    :param mp3_path: MP3文件的路径（相对/绝对）
    """
    # 1. 初始化pygame音频模块（必须步骤，否则无法加载/播放音频）
    pygame.mixer.init()
    
    try:
        # 2. 加载MP3文件（核心：将音频文件读入内存）
        pygame.mixer.music.load(mp3_path)
        print(f"✅ 成功加载MP3文件：{mp3_path}")
        print("▶️  开始播放歌曲...（播放完成后程序自动结束）")
        
        # 3. 开始播放（关键修正：参数为loops=0，而非loop）
        # loops=0：播放1次；loops=-1：无限循环播放（按Ctrl+C停止）
        pygame.mixer.music.play(loops=0)
        
        # 4. 持续检测播放状态，直到歌曲播放完成（避免程序提前退出）
        # pygame.mixer.music.get_busy()：返回True表示音频正在播放，False表示播放完成
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)  # 每0.1秒检测一次，几乎不占用系统资源
        
        print("⏹️  MP3文件播放完成！")

    # 错误处理1：文件找不到（最常见问题）
    except FileNotFoundError:
        print(f"❌ 错误：未找到MP3文件「{mp3_path}」")
        print("   排查建议：")
        print("   1. 检查文件路径是否正确（相对路径需确保MP3和代码同目录）；")
        print("   2. 文件名/后缀是否拼写正确（如.mp3不要写成.MP3或.m3u）；")
        print("   3. 路径中不要有中文/特殊符号（建议改为纯英文/数字）。")

    # 错误处理2：音频格式不支持/文件损坏
    except pygame.error as e:
        print(f"❌ 播放失败：{e}")
        print("   排查建议：")
        print("   1. 确保文件是标准MP3格式（避免无损/加密音频）；")
        print("   2. 检查MP3文件是否损坏（用播放器手动播放测试）。")

    # 错误处理3：其他未知错误
    except Exception as e:
        print(f"❌ 未知错误：{str(e)}")

    # 最终步骤：无论是否报错，释放音频资源（避免占用系统资源）
    finally:
        pygame.mixer.music.stop()  # 停止播放（若有残留）
        pygame.mixer.quit()        # 退出音频模块

# 程序入口：直接运行该文件时执行播放函数
if __name__ == "__main__":
    play_mp3_file(MP3_FILE_PATH)