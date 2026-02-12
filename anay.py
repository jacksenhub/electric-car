import pandas as pd
import numpy as np
import os

# ==========================================
# 1. 配置与 Excel 数据加载
# ==========================================
FILE_NAME = 'DrivingData.xlsx'  # 确保文件名准确

if not os.path.exists(FILE_NAME):
    print(f"❌ 错误：在当前目录下找不到 {FILE_NAME}")
    print(f"当前目录文件列表：{os.listdir('.')}")
else:
    print(f"🚀 正在从 Excel 读取 Activity 表（17805 辆车数据）...")
    # 指定读取 'Activity' 工作表
    try:
        df = pd.read_excel(FILE_NAME, sheet_name='Activity')
    except Exception as e:
        print(f"⚠️ 无法读取工作表 'Activity'，尝试读取第一个工作表。错误: {e}")
        df = pd.read_excel(FILE_NAME)

    # ==========================================
    # 2. 数据清洗与特征计算
    # ==========================================
    # 计算持续时间 (小时)
    df['Duration'] = df['End time (hour)'] - df['Start time (hour)']

    print("📊 正在进行全量数据统计分析...")

    # 分类别提取数据
    driving_df = df[df['State'] == 'Driving']
    charging_df = df[df['State'] == 'Charging']
    parking_df = df[df['State'] == 'Parked']

    # 按车辆聚合核心指标
    vehicle_stats = df.groupby('Vehicle ID').agg(
        日总行驶里程=('Distance (mi)', lambda x: x[x > 0].sum()),
        日行驶总时长=('Duration', lambda x: x[df.loc[x.index, 'State'] == 'Driving'].sum()),
        日停车总时长=('Duration', lambda x: x[df.loc[x.index, 'State'] == 'Parked'].sum()),
        日充电总时长=('Duration', lambda x: x[df.loc[x.index, 'State'] == 'Charging'].sum()),
        日出行次数=('State', lambda x: (x == 'Driving').sum())
    ).fillna(0)

    # ==========================================
    # 3. 输出描述性统计表格 (Table 1)
    # ==========================================
    desc_table = vehicle_stats.describe().T[['mean', 'std', 'min', '50%', 'max']]
    desc_table.columns = ['均值', '标准差', '最小值', '中位数', '最大值']
    
    print("\n" + "="*50)
    print("📋 表1：电动汽车集群行为描述性统计 (Descriptive Statistics)")
    print("="*50)
    print(desc_table.to_string())
    desc_table.to_csv('EV_Mobility_Description_Stats.csv', encoding='utf-8-sig')

    # ==========================================
    # 4. 全天出行/充电概率分布 (24h 序列)
    # ==========================================
    print("\n⏳ 正在计算 24 小时概率分布曲线（步长：0.5h）...")
    
    time_bins = np.arange(0, 24.5, 0.5)
    total_vehicles = df['Vehicle ID'].nunique()
    
    prob_list = []
    for t in time_bins:
        # 统计在 t 时刻处于某种状态的车辆数 (Start <= t < End)
        active_rows = df[(df['Start time (hour)'] <= t) & (df['End time (hour)'] > t)]
        counts = active_rows['State'].value_counts()
        
        prob_list.append({
            '时刻(Hour)': t,
            '出行概率(Driving)': counts.get('Driving', 0) / total_vehicles,
            '充电概率(Charging)': counts.get('Charging', 0) / total_vehicles,
            '停车概率(Parked)': counts.get('Parked', 0) / total_vehicles
        })

    prob_df = pd.DataFrame(prob_list)
    prob_df.to_csv('EV_Time_Probability_Distribution.csv', index=False, encoding='utf-8-sig')

    print("\n" + "="*50)
    print("✅ 分析全流程完成！")
    print(f"1. 核心统计表已保存至: EV_Mobility_Description_Stats.csv")
    print(f"2. 24h 概率分布数据已保存至: EV_Time_Probability_Distribution.csv")
    print("="*50)

    # ==========================================
    # 5. 简要科研结论
    # ==========================================
    peak_drive = prob_df.loc[prob_df['出行概率(Driving)'].idxmax()]
    peak_charge = prob_df.loc[prob_df['充电概率(Charging)'].idxmax()]
    
    print(f"💡 结论摘要：")
    print(f"• 集群出行高峰出现在 {peak_drive['时刻(Hour)']}h，此时约 {peak_drive['出行概率(Driving)']:.2%} 的车辆在路上。")
    print(f"• 集群充电高峰出现在 {peak_charge['时刻(Hour)']}h，此时约 {peak_charge['充电概率(Charging)']:.2%} 的车辆正在充电。")
    print(f"• 车辆日均静止（停车）时长高达 {vehicle_stats['日停车总时长'].mean():.2f} 小时，具有极强的 V2G 调控潜力。")