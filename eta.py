import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==========================================
# 1. 充电负荷计算与初始化
# ==========================================

# 文件路径（根据你的实际位置修改）
file_path = r"E:\2023210119贾正鑫\DrivingData_20EVs.xlsx"

try:
    # 读取 Excel 的两个工作表
    df_init = pd.read_excel(file_path, sheet_name='Initialization')
    df_activity = pd.read_excel(file_path, sheet_name='Activity')
    print(">>> 数据加载成功！")
except Exception as e:
    print(f">>> 加载失败: {e}")
    # 如果是在上传环境，尝试读取CSV格式
    df_init = pd.read_csv("DrivingData_20EVs.xlsx - Initialization.csv")
    df_activity = pd.read_csv("DrivingData_20EVs.xlsx - Activity.csv")

# 参数设置
num_evs = 20
time_step = 0.25  # 15分钟 = 0.25小时
time_slots = 96   # 一天 24/0.25 = 96 个点
time_index = np.linspace(0, 24, time_slots, endpoint=False)

# 初始化功率矩阵 (车辆x时间) 和 可充电窗口矩阵
P_baseline = np.zeros((num_evs, time_slots))
Availability = np.zeros((num_evs, time_slots))

# 从 Activity 表中提取每辆车的最大充电功率（取第一条记录中的 P_max）
P_max_vec = np.zeros(num_evs)
for vid in range(1, num_evs + 1):
    car_act = df_activity[df_activity['Vehicle ID'] == vid]
    if len(car_act) > 0:
        # 取该车所有记录中的 P_max（应该都相同）
        p_max_val = car_act['P_max (W)'].iloc[0]
        if p_max_val > 0:
            P_max_vec[vid-1] = p_max_val
        else:
            P_max_vec[vid-1] = 1440  # 默认值，如果为 -1 则使用 1440

# 遍历每辆车，提取原始充电负荷和停车窗口
for vid in range(1, num_evs + 1):
    car_act = df_activity[df_activity['Vehicle ID'] == vid]
    
    for _, row in car_act.iterrows():
        # 将小时转换为索引
        start_idx = int(row['Start time (hour)'] / time_step)
        end_idx = int(row['End time (hour)'] / time_step)
        end_idx = min(end_idx, time_slots)
        
        # [Step 1] 累加原始充电功率 (Baseline)
        if row['State'] == 'Charging':
            P_baseline[vid-1, start_idx:end_idx] = P_max_vec[vid-1]
        
        # [Step 3 约束准备] 确定可充电时段：在家(Home)且非驾驶(Driving)状态
        if row['Location'] == 'Home' and row['State'] != 'Driving':
            Availability[vid-1, start_idx:end_idx] = 1

# 计算各时段总负荷 (单位：W)
total_load_base = np.sum(P_baseline, axis=0)

# ==========================================
# 2. 波动率量化指标定义
# ==========================================

def calc_volatility(load_series):
    """定义波动率为负荷的方差"""
    return np.var(load_series)

vol_base = calc_volatility(total_load_base)

# ==========================================
# 3. 优化建模与求解
# ==========================================

# 目标电量约束：优化后的充电电量必须等于原始充电电量 (kWh)
target_energy_kwh = np.sum(P_baseline, axis=1) * time_step

# 优化目标函数：最小化总负荷方差
def objective(x):
    # x 是展平后的功率数组 (num_evs * time_slots)
    x_reshaped = x.reshape((num_evs, time_slots))
    total_load = np.sum(x_reshaped, axis=0)
    return np.var(total_load)

# 约束条件列表
constraints = []
for i in range(num_evs):
    # 每辆车的能量守恒约束
    def energy_cons(x, idx=i):
        x_reshaped = x.reshape((num_evs, time_slots))
        return np.sum(x_reshaped[idx]) * time_step - target_energy_kwh[idx]
    
    constraints.append({'type': 'eq', 'fun': energy_cons})

# 边界条件：限制充电时间范围和单车最大功率
bounds = []
for i in range(num_evs):
    for t in range(time_slots):
        if Availability[i, t] == 1:
            bounds.append((0, P_max_vec[i]))  # 在家可充，功率为 0 到 P_max
        else:
            bounds.append((0, 0))            # 不在家，功率强制为 0

print(">>> 正在优化负荷分布，请稍候...")
# 初始值设为基线负荷
res = minimize(objective, P_baseline.flatten(), method='SLSQP', 
               bounds=bounds, constraints=constraints, options={'maxiter': 50})

P_optimized = res.x.reshape((num_evs, time_slots))
total_load_opt = np.sum(P_optimized, axis=0)
vol_opt = calc_volatility(total_load_opt)

# ==========================================
# 4. 结果对比分析与可视化
# ==========================================

print("\n" + "="*30)
print(f"{'指标':<15} | {'优化前 (Base)':<12} | {'优化后 (Opt)':<12}")
print("-"*45)
print(f"{'波动率(方差)':<15} | {vol_base:<12.2f} | {vol_opt:<12.2f}")
print(f"{'峰值负荷(W)':<15} | {np.max(total_load_base):<12.2f} | {np.max(total_load_opt):<12.2f}")
print(f"{'峰谷差(W)':<15} | {np.max(total_load_base)-np.min(total_load_base):<12.2f} | {np.max(total_load_opt)-np.min(total_load_opt):<12.2f}")
print("="*30)

# 绘图
plt.rcParams['font.sans-serif'] = ['SimHei'] # 解决中文显示问题
plt.figure(figsize=(12, 6))

plt.plot(time_index, total_load_base, color='red', linestyle='--', label='优化前 (无序充电)')
plt.plot(time_index, total_load_opt, color='green', linewidth=2, label='优化后 (有序充电)')
plt.fill_between(time_index, total_load_opt, color='green', alpha=0.15)

plt.title('20辆电动汽车充电负荷优化对比', fontsize=14)
plt.xlabel('时间 (24小时制)', fontsize=12)
plt.ylabel('总充电功率 (W)', fontsize=12)
plt.xticks(np.arange(0, 25, 2))
plt.grid(alpha=0.3)
plt.legend()

plt.show()