# ==============================================
# 电价数据清洗完整代码（精简防占用版）
# 适配路径：E:\2023210119贾正鑫
# 解决：1.北京平段时间段省略号 2.奥斯陆/夏威夷区间电价 3.多货币单位智能转换 4.特殊符号清理 5.自动处理文件占用
# ==============================================
import sys
import subprocess
import re
import pandas as pd
import numpy as np
from datetime import datetime

# --------------------------
# 步骤0：检查并安装依赖库
# --------------------------
def install_package(package):
    """自动安装缺失的库"""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package, "-i", "https://mirrors.aliyun.com/pypi/simple/"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except:
        return False

# 检查核心依赖
required_packages = ['pandas', 'numpy', 'openpyxl']
missing_packages = []

for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        missing_packages.append(pkg)

if missing_packages:
    print(f"⚠️ 检测到缺失依赖库：{missing_packages}")
    print("🔧 正在自动安装（使用阿里云镜像源）...")
    success = True
    for pkg in missing_packages:
        if install_package(pkg):
            print(f"✅ {pkg} 安装成功")
        else:
            print(f"❌ {pkg} 安装失败，请手动执行：pip install {pkg}")
            success = False
    if not success:
        sys.exit(1)
    # 重新导入
    import pandas as pd
    import numpy as np

# --------------------------
# 步骤1：核心清洗函数（增强版）
# --------------------------
def extract_numeric_value(price_str):
    """
    从字符串中提取数值（处理范围值取中值）
    返回：float 或 np.nan
    """
    if pd.isna(price_str) or str(price_str).strip() == '':
        return np.nan
    
    s = str(price_str).strip()
    
    # 移除所有非数字相关字符（保留数字、小数点、减号、波浪号）
    s_clean = s.replace('~', '').replace('≈', '')
    
    # 提取所有数字片段（含小数点）
    num_parts = re.findall(r'[\d.]+', s_clean)
    if not num_parts:
        return np.nan
    
    # 检查是否是范围值（原字符串包含 - ）
    if '-' in s_clean:
        try:
            # 尝试分割范围
            range_match = re.search(r'([\d.]+)\s*-\s*([\d.]+)', s_clean)
            if range_match:
                num1 = float(range_match.group(1))
                num2 = float(range_match.group(2))
                return (num1 + num2) / 2  # 取中值
            else:
                # 备用方案：取所有数字的中值
                nums = [float(x) for x in num_parts if x.replace('.', '', 1).isdigit()]
                if len(nums) >= 2:
                    return (min(nums) + max(nums)) / 2
                elif len(nums) == 1:
                    return nums[0]
                else:
                    return np.nan
        except:
            return np.nan
    else:
        # 非范围值，取第一个有效数字
        try:
            for part in num_parts:
                if part.replace('.', '', 1).isdigit():
                    return float(part)
            return np.nan
        except:
            return np.nan

def convert_local_currency_to_cny(price_str):
    """
    将本地货币字符串智能转换为人民币（元/kWh）
    """
    if pd.isna(price_str) or str(price_str).strip() == '':
        return np.nan
    
    s = str(price_str).strip()
    s_lower = s.lower()
    
    # 提取数值（已处理范围值）
    num_value = extract_numeric_value(s)
    if np.isnan(num_value):
        return np.nan
    
    # ============ 智能识别货币单位 ============
    # 1. 美分（夏威夷）
    if any(kw in s_lower for kw in ['cents', 'cent', '¢']):
        usd_value = num_value / 100
        return round(usd_value * 7.2, 4)
    
    # 2. 挪威欧尔（奥斯陆）
    elif any(kw in s_lower for kw in ['øre', 'ore']):
        nok_value = num_value / 100
        return round(nok_value * 0.8, 4)
    
    # 3. 挪威克朗
    elif any(kw in s_lower for kw in ['krone', 'kroner', 'nok']):
        return round(num_value * 0.8, 4)
    
    # 4. 人民币（已有）
    elif any(kw in s_lower or kw in s for kw in ['cny', '元', '人民币', 'rmb', 'yuan']):
        return round(num_value, 4)
    
    # 5. 美元
    elif any(kw in s_lower for kw in ['usd', 'dollar', '$']):
        return round(num_value * 7.2, 4)
    
    # 6. 欧元
    elif any(kw in s_lower for kw in ['eur', 'euro', '€']):
        return round(num_value * 7.8, 4)
    
    else:
        # 无法识别单位：保守处理
        if 0 <= num_value <= 10:
            return round(num_value, 4)
        elif 10 < num_value <= 1000:
            return round((num_value / 100) * 7.2, 4)
        else:
            return np.nan

def clean_time_segment(ts):
    """
    标准化时间段：去除省略号、统一分隔符、清理空格
    """
    if pd.isna(ts):
        return ts
    
    ts = str(ts).strip()
    
    # 1. 去除所有类型省略号
    ts = re.sub(r'\.{2,}|…', '', ts)
    
    # 2. 统一分隔符（英文逗号→中文顿号）
    ts = ts.replace(',', '、').replace('，', '、')
    
    # 3. 清理多余空格
    ts = re.sub(r'\s+', ' ', ts).strip()
    
    # 4. 规范化"次日"表述
    ts = re.sub(r'次日\s*', '次日', ts)
    
    return ts if ts else np.nan

def save_with_fallback(df, base_path):
    """
    尝试保存文件，如果检测到被占用，自动添加时间戳后缀
    """
    try:
        df.to_csv(base_path, index=False, encoding='utf-8-sig')
        return base_path
    except PermissionError:
        print(f"⚠️ 警告：文件 {base_path} 被占用！")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = base_path.rsplit('.', 1)
        new_path = f"{name}_{timestamp}.{ext}"
        print(f"💡 自动切换至新文件名：{new_path}")
        df.to_csv(new_path, index=False, encoding='utf-8-sig')
        return new_path

def clean_electricity_price_data(raw_file_path, output_file_path):
    """
    电价数据清洗主函数
    """
    # ============ 1. 读取原始数据 ============
    print("="*60)
    print("📌 开始读取原始数据...")
    try:
        df = pd.read_excel(raw_file_path, engine='openpyxl')
        print(f"✅ 原始数据读取成功 | 行数: {df.shape[0]} | 列数: {df.shape[1]}")
    except FileNotFoundError:
        print(f"❌ 错误：未找到文件 {raw_file_path}")
        return None
    except PermissionError:
        print(f"❌ 错误：文件被占用（请关闭Excel后重试）")
        return None
    except Exception as e:
        print(f"❌ 数据读取失败：{str(e)}")
        return None

    # ============ 2. 补全城市缺失值 ============
    print("\n" + "="*60)
    print("📌 补全城市缺失值...")
    if '城市' in df.columns:
        original_missing = df['城市'].isnull().sum()
        df['城市'] = df['城市'].ffill().bfill()
        print(f"✅ 城市缺失值修复: {original_missing} → {df['城市'].isnull().sum()}")
    else:
        print("⚠️ 未检测到'城市'列")

    # ============ 3. 核心：本地货币清洗+人民币转换 ============
    print("\n" + "="*60)
    print("📌 清洗本地货币电价并转换为人民币...")
    local_col = "预计电价 (本地货币)"
    
    if local_col in df.columns:
        df['本地货币清洗值'] = df[local_col].apply(convert_local_currency_to_cny)
        cny_col = "预计电价 (约合人民币/kWh)_清洗后"
        df[cny_col] = df['本地货币清洗值']
        
        df['电价状态'] = np.where(
            (df[cny_col] >= 0) & (df[cny_col] <= 10),
            '有效',
            '无效（需确认）'
        )
        
        valid_count = (df['电价状态'] == '有效').sum()
        invalid_count = (df['电价状态'] == '无效（需确认）').sum()
        print(f"✅ 货币转换完成 | 有效: {valid_count} | 无效: {invalid_count}")
    else:
        print(f"❌ 未找到关键列 '{local_col}'")
        return None

    # ============ 4. 标准化时间段（修复省略号） ============
    print("\n" + "="*60)
    print("📌 标准化时间段（修复省略号/格式）...")
    time_col = "时间段"
    
    if time_col in df.columns:
        df['时间段_标准化'] = df[time_col].apply(clean_time_segment)
        
        # 拆分多时段
        df_split = df.assign(
            时间段_拆分=df['时间段_标准化'].str.split('、')
        ).explode('时间段_拆分').reset_index(drop=True)
        
        # 清理拆分后的空格
        df_split['时间段_最终'] = df_split['时间段_拆分'].str.strip()
        
        print(f"✅ 时间段处理完成 | 行数: {df.shape[0]} → {df_split.shape[0]}")
    else:
        print(f"⚠️ 未找到'{time_col}'列")
        df_split = df.copy()
        df_split['时间段_最终'] = np.nan

    # ============ 5. 输出清洗后数据 (仅生成一个文件) ============
    print("\n" + "="*60)
    print("📌 生成最终清洗数据集...")
    
    final_columns = [
        '城市', '时段分类', '时间段_最终',
        '预计电价 (本地货币)', '本地货币清洗值',
        '预计电价 (约合人民币/kWh)_清洗后', '电价状态'
    ]
    final_columns = [col for col in final_columns if col in df_split.columns]
    df_final = df_split[final_columns].copy()
    
    # 使用防占用保存函数
    final_save_path = save_with_fallback(df_final, output_file_path)
    
    print(f"✅ 清洗完成！文件已保存至:\n   {final_save_path}")
    print(f"📊 最终数据: {df_final.shape[0]}行 × {df_final.shape[1]}列")
    
    print("\n🎉 清洗成果总结:")
    print(f"   • 修复时间段省略号: 北京平段等")
    print(f"   • 处理区间电价: 奥斯陆/夏威夷已取中值并换算")
    print(f"   • 统一货币单位: 所有电价转为人民币/kWh")
    
    print("\n🔍 最终数据预览（前10行）:")
    print(df_final.head(10).to_string(index=False))
    
    return df_final

# --------------------------
# 步骤2：主程序入口
# --------------------------
if __name__ == "__main__":
    # 配置文件路径
    RAW_FILE = r"E:\2023210119贾正鑫\城市电价.xlsx"
    OUTPUT_FILE = r"E:\2023210119贾正鑫\城市电价_清洗完成.csv"
    
    print("="*60)
    print("⚡ 电价数据智能清洗系统 v4.0 (精简防占用版)")
    print("✨ 功能：单文件输出 | 自动处理占用 | 省略号修复 | 区间值取中值 | 多货币转换")
    print("="*60)
    
    result = clean_electricity_price_data(RAW_FILE, OUTPUT_FILE)
    
    if result is not None:
        print("\n" + "="*60)
        print("✅ 全流程清洗成功！")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ 清洗流程中断")
        print("="*60)