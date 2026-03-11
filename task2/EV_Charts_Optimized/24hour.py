import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO

# ==========================================
# 1. 数据准备 (直接嵌入数据，无需外部CSV即可运行)
# ==========================================

time_data_str = """时刻(Hour),出行概率(Driving),充电概率(Charging),停车概率(Parked)
0.0,0.0023,0.94799,0.04971
0.5,0.00315,0.94912,0.04774
1.0,0.00152,0.95164,0.04684
1.5,0.0023,0.95226,0.04544
2.0,0.00163,0.95372,0.04465
2.5,0.00124,0.95439,0.04437
3.0,0.00067,0.95535,0.04398
3.5,0.00084,0.95569,0.04347
4.0,0.00371,0.95411,0.04218
4.5,0.00871,0.94861,0.04268
5.0,0.01488,0.9412,0.04392
5.5,0.02555,0.92732,0.04712
6.0,0.03291,0.91446,0.05263
6.5,0.05594,0.88666,0.0574
7.0,0.07397,0.85779,0.06824
7.5,0.103,0.81404,0.08295
8.0,0.10396,0.78602,0.11003
8.5,0.08571,0.78034,0.13395
9.0,0.08767,0.75973,0.1526
9.5,0.08234,0.7449,0.17276
10.0,0.09239,0.71738,0.19023
10.5,0.08099,0.70666,0.21236
11.0,0.09194,0.68986,0.2182
11.5,0.09009,0.6847,0.22522
12.0,0.09857,0.67526,0.22617
12.5,0.09093,0.68385,0.22522
13.0,0.09503,0.69076,0.21421
13.5,0.09014,0.69773,0.21213
14.0,0.1002,0.69638,0.20343
14.5,0.10154,0.69773,0.20073
15.0,0.11553,0.6975,0.18697
15.5,0.11193,0.70997,0.1781
16.0,0.12283,0.71441,0.16276
16.5,0.11643,0.72878,0.15479
17.0,0.13238,0.72109,0.14653
17.5,0.12058,0.73564,0.14378
18.0,0.10385,0.74906,0.14709
18.5,0.08093,0.76574,0.15333
19.0,0.06728,0.77871,0.154
19.5,0.0474,0.80348,0.14912
20.0,0.04044,0.82393,0.13564
20.5,0.03241,0.84802,0.11957
21.0,0.03061,0.87313,0.09627
21.5,0.02539,0.89907,0.07554
22.0,0.02044,0.91783,0.06172
22.5,0.01393,0.93328,0.05279
23.0,0.00938,0.94204,0.04858
23.5,0.00517,0.94883,0.046
24.0,0.0,0.0,0.0"""

stats_data_str = """,均值,标准差,最小值,中位数,最大值
日总行驶里程,34.0448,41.35487,1.11111,23.0,1400.0
日行驶总时长,1.2313,0.93065,0.03333,1.0,16.5
日停车总时长,2.92808,3.99493,0.0,1.5,23.85
日充电总时长,19.84062,4.25605,0.0,21.26667,23.93333
日出行次数,4.19034,2.37287,1.0,4.0,25.0"""

# 读取数据
time_dist = pd.read_csv(StringIO(time_data_str))
stats = pd.read_csv(StringIO(stats_data_str), index_col=0)

# 设置绘图风格和中文字体
sns.set_style("whitegrid")
# 尝试加载常见中文字体，防止乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Microsoft YaHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# 创建输出目录
output_dir = 'EV_Charts_Optimized'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 定义优化后的高对比度配色方案
COLOR_DRIVING = '#E74C3C'  # 深红
COLOR_CHARGING = '#1ABC9C' # 鲜明青绿
COLOR_PARKED = '#3498DB'   # 宝蓝

# ==========================================
# 图表 1: 24小时三状态概率分布 (颜色优化版)
# ==========================================
fig1, ax1 = plt.subplots(figsize=(12, 6))

# 绘制线条 - 加粗至 3.0
ax1.plot(time_dist['时刻(Hour)'], time_dist['出行概率(Driving)'], 
         label='出行 (Driving)', color=COLOR_DRIVING, linewidth=3)
ax1.plot(time_dist['时刻(Hour)'], time_dist['充电概率(Charging)'], 
         label='充电 (Charging)', color=COLOR_CHARGING, linewidth=3)
ax1.plot(time_dist['时刻(Hour)'], time_dist['停车概率(Parked)'], 
         label='停车 (Parked)', color=COLOR_PARKED, linewidth=3)

# 填充区域 - 降低透明度至 0.15，避免干扰线条
ax1.fill_between(time_dist['时刻(Hour)'], time_dist['出行概率(Driving)'], alpha=0.15, color=COLOR_DRIVING)
ax1.fill_between(time_dist['时刻(Hour)'], time_dist['充电概率(Charging)'], alpha=0.15, color=COLOR_CHARGING)
ax1.fill_between(time_dist['时刻(Hour)'], time_dist['停车概率(Parked)'], alpha=0.15, color=COLOR_PARKED)

ax1.set_title('图1: 电动汽车24小时三状态概率分布 (高对比度版)', fontsize=16, fontweight='bold', pad=15)
ax1.set_xlabel('时刻 (小时)', fontsize=12)
ax1.set_ylabel('概率', fontsize=12)
# 图例加黑框，更清晰
ax1.legend(loc='upper right', fontsize=11, frameon=True, edgecolor='black', facecolor='white')
ax1.set_xlim(0, 24)
ax1.set_ylim(0, 1)
ax1.grid(True, linestyle='--', alpha=0.6)

plt.savefig(f'{output_dir}/01_24h_State_Probability_Optimized.png', dpi=300, bbox_inches='tight')
plt.close(fig1)
print(f"✅ 图表1已保存: {output_dir}/01_24h_State_Probability_Optimized.png")

# ==========================================
# 图表 2: 描述性统计条形图
# ==========================================
fig2, ax2 = plt.subplots(figsize=(10, 6))
metrics = stats.index.tolist()
means = stats['均值'].values
stds = stats['标准差'].values
# 为条形图分配协调的颜色
bar_colors = [COLOR_DRIVING, COLOR_CHARGING, COLOR_PARKED, '#95A5A6', '#F1C40F']

bars = ax2.bar(metrics, means, yerr=stds, color=bar_colors, capsize=6, edgecolor='black', alpha=0.9, linewidth=1.2)
ax2.set_title('图2: 电动汽车移动性关键指标统计 (均值±标准差)', fontsize=16, fontweight='bold', pad=15)
ax2.set_ylabel('数值', fontsize=12)
ax2.tick_params(axis='x', rotation=15, labelsize=11)
ax2.grid(True, axis='y', linestyle='--', alpha=0.6)

# 添加数值标签
for i, v in enumerate(means):
    offset = stds[i] + (max(means) * 0.05)
    ax2.text(i, v + offset, f'{v:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.savefig(f'{output_dir}/02_Descriptive_Stats_Bar.png', dpi=300, bbox_inches='tight')
plt.close(fig2)
print(f"✅ 图表2已保存: {output_dir}/02_Descriptive_Stats_Bar.png")

# ==========================================
# 图表 3: 早晚高峰出行概率对比
# ==========================================
fig3, ax3 = plt.subplots(figsize=(8, 6))
morning_mask = (time_dist['时刻(Hour)'] >= 6) & (time_dist['时刻(Hour)'] <= 9)
evening_mask = (time_dist['时刻(Hour)'] >= 17) & (time_dist['时刻(Hour)'] <= 20)

morning_avg = time_dist.loc[morning_mask, '出行概率(Driving)'].mean()
evening_avg = time_dist.loc[evening_mask, '出行概率(Driving)'].mean()

labels = ['早高峰\n(06:00-09:00)', '晚高峰\n(17:00-20:00)']
values = [morning_avg, evening_avg]
# 使用出行颜色的深浅变体
bar_colors_peak = ['#E67E22', COLOR_DRIVING] 

bars = ax3.bar(labels, values, color=bar_colors_peak, edgecolor='black', alpha=0.9, linewidth=1.2)
ax3.set_title('图3: 早晚高峰平均出行概率对比', fontsize=16, fontweight='bold', pad=15)
ax3.set_ylabel('平均出行概率', fontsize=12)
ax3.set_ylim(0, max(values) * 1.2)
ax3.grid(True, axis='y', linestyle='--', alpha=0.6)

for i, v in enumerate(values):
    ax3.text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold', color='#333')

plt.savefig(f'{output_dir}/03_Peak_Hours_Comparison.png', dpi=300, bbox_inches='tight')
plt.close(fig3)
print(f"✅ 图表3已保存: {output_dir}/03_Peak_Hours_Comparison.png")

# ==========================================
# 图表 4: 充电概率时间分布散点热力图
# ==========================================
fig4, ax4 = plt.subplots(figsize=(12, 6))
hours = time_dist['时刻(Hour)'].values
charging_prob = time_dist['充电概率(Charging)'].values

# 使用与充电状态一致的青色系 colormap
scatter = ax4.scatter(hours, charging_prob, c=charging_prob, cmap='Greens', 
                      s=150, edgecolors='black', alpha=0.85, linewidth=1.2)

ax4.set_title('图4: 24小时充电概率分布热力趋势', fontsize=16, fontweight='bold', pad=15)
ax4.set_xlabel('时刻 (小时)', fontsize=12)
ax4.set_ylabel('充电概率', fontsize=12)
ax4.set_xlim(-0.5, 24.5)
ax4.set_ylim(0.6, 1.0) 
ax4.grid(True, linestyle='--', alpha=0.6)

cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('概率值', fontsize=11)

# 标注最高点
max_idx = charging_prob.argmax()
ax4.annotate(f'峰值: {charging_prob[max_idx]:.2f}\n({int(hours[max_idx])}:00)', 
             xy=(hours[max_idx], charging_prob[max_idx]), 
             xytext=(hours[max_idx]+1, charging_prob[max_idx]-0.05),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red', fontweight='bold')

plt.savefig(f'{output_dir}/04_Charging_Probability_Heatmap.png', dpi=300, bbox_inches='tight')
plt.close(fig4)
print(f"✅ 图表4已保存: {output_dir}/04_Charging_Probability_Heatmap.png")

print(f"\n🎉 所有图表已生成并保存至 '{output_dir}' 文件夹！")
print("💡 提示：图表1已特别优化颜色对比度，三条线现在清晰可辨。")