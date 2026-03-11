# 导入Windows内置库，无需pip安装任何依赖
import winsound
import time

# 定义8个不同音调的频率（对应“哆来咪发索拉西高哆”，覆盖常用音域）
# 频率越高，音调越高，可根据需求微调数值
NOTE_FREQUENCIES = [
    261,  # 第1个音调：C4 哆
    294,  # 第2个音调：D4 来
    330,  # 第3个音调：E4 咪
    349,  # 第4个音调：F4 发
    392,  # 第5个音调：G4 索
    440,  # 第6个音调：A4 拉
    494,  # 第7个音调：B4 西
    523   # 第8个音调：C5 高哆
]

# 核心参数（完全匹配你的需求，无需修改）
TONE_DURATION_MS = 1000  # 每个音调播放时长：1000毫秒 = 1秒
INTERVAL_DURATION_S = 0.5  # 音调之间的间隔：0.5秒

def play_8_tones_cycle():
    """
    核心功能：循环播放8个音调
    - 循环逻辑：遍历8个频率，逐个播放
    - 时长控制：每个音调播放1秒，间隔0.5秒（最后一个无间隔）
    """
    print("===== 开始播放8个音调（每个1秒，间隔0.5秒）=====")
    
    # 循环遍历8个音调频率（核心循环语句）
    for tone_index, freq in enumerate(NOTE_FREQUENCIES):
        # 计算当前音调序号（1-8），方便日志输出
        current_tone = tone_index + 1
        
        # 播放当前音调：winsound.Beep(频率, 时长) 是Windows原生蜂鸣函数
        print(f"正在播放第 {current_tone} 个音调 | 频率：{freq}Hz（播放1秒）")
        winsound.Beep(freq, TONE_DURATION_MS)
        
        # 最后一个音调播放完成后，不执行间隔等待
        if current_tone < len(NOTE_FREQUENCIES):
            print(f"第 {current_tone} 个音调播放完成，等待 {INTERVAL_DURATION_S} 秒\n")
            time.sleep(INTERVAL_DURATION_S)  # 间隔0.5秒
    
    print("\n===== 8个音调全部播放完成 =====")

# 程序入口：直接运行该文件时执行核心函数
if __name__ == "__main__":
    play_8_tones_cycle()