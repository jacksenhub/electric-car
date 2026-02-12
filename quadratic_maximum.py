"""
二次函数极大值求解：解析法、数值法与可视化
函数：y = -x² + 4x + 5
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# ==================== 第一部分：解析法推导 ====================
# 定义常量
PLOT_WIDTH = 16
PLOT_HEIGHT = 12
FIGURE_DPI = 300
X_MIN, X_MAX = -3, 7
Y_MIN, Y_MAX = -10, 12
DEFAULT_SEED = 42
np.random.seed(DEFAULT_SEED)

print("=" * 60)
print("第一部分：解析法推导")
print("=" * 60)

def f(x):
    """定义目标函数：y = -x² + 4x + 5"""
    return -x**2 + 4*x + 5

def f_prime(x):
    """一阶导数：y' = -2x + 4"""
    return -2*x + 4

def f_double_prime(x):
    """二阶导数：y'' = -2"""
    return -2

print("1. 函数定义：")
print("   y = -x² + 4x + 5")
print("   y' = -2x + 4")
print("   y'' = -2\n")

print("2. 求驻点：")
print("   令 y' = 0")
print("   -2x + 4 = 0")
print("   x = 2\n")

print("3. 判断极值类型：")
print("   y'' = -2 < 0")
print("   二阶导数为负，说明该点为极大值点\n")

print("4. 计算极大值：")
print("   将 x = 2 代入原函数：")
print("   y = -(2)² + 4×2 + 5")
print("     = -4 + 8 + 5")
print("     = 9\n")

print("5. 解析法结果：")
print("   极大值点：x = 2")
print("   极大值：y = 9")
print("=" * 60)

# ==================== 第二部分：数值法求解 ====================
print("\n" + "=" * 60)
print("第二部分：数值法求解（使用SciPy）")
print("=" * 60)

def negative_f(x):
    """负函数，用于最小化求最大值"""
    return -f(x)

# 方法1：使用minimize函数
print("方法1：使用scipy.optimize.minimize")
initial_guess = 0  # 初始猜测值
bounds = [(-10, 10)]  # 边界条件

result1 = optimize.minimize(negative_f, initial_guess, bounds=bounds, method='L-BFGS-B')
if result1.success:
    x_max1 = result1.x[0]
    y_max1 = f(x_max1)
    print(f"  成功收敛")
    print(f"  极大值点: x = {x_max1:.10f}")
    print(f"  极大值: y = {y_max1:.10f}")
    print(f"  迭代次数: {result1.nit}")
else:
    print(f"  优化失败: {result1.message}")

# 方法2：使用minimize_scalar函数
print("\n方法2：使用scipy.optimize.minimize_scalar")
result2 = optimize.minimize_scalar(negative_f, bounds=(-10, 10), method='bounded')
x_max2 = result2.x
y_max2 = f(x_max2)
print(f"  极大值点: x = {x_max2:.10f}")
print(f"  极大值: y = {y_max2:.10f}")
print(f"  函数调用次数: {result2.nfev}")

# 方法3：使用fminbound（备选方法）
print("\n方法3：使用scipy.optimize.fminbound")
x_max3 = optimize.fminbound(negative_f, -10, 10)
y_max3 = f(x_max3)
print(f"  极大值点: x = {x_max3:.10f}")
print(f"  极大值: y = {y_max3:.10f}")

# 结果对比
print("\n" + "-" * 60)
print("数值法与解析法结果对比：")
print(f"{'方法':<25} {'x值':<15} {'y值':<15} {'与解析法x误差':<15}")
print("-" * 60)
print(f"{'解析法':<25} {2:<15.10f} {9:<15.10f} {0:<15.10f}")
print(f"{'minimize':<25} {x_max1:<15.10f} {y_max1:<15.10f} {abs(x_max1-2):<15.10f}")
print(f"{'minimize_scalar':<25} {x_max2:<15.10f} {y_max2:<15.10f} {abs(x_max2-2):<15.10f}")
print(f"{'fminbound':<25} {x_max3:<15.10f} {y_max3:<15.10f} {abs(x_max3-2):<15.10f}")

print("=" * 60)

# ==================== 第三部分：可视化 ====================
print("\n" + "=" * 60)
print("第三部分：结果可视化")
print("=" * 60)

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig = plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
fig.suptitle('二次函数极大值求解：y = -x² + 4x + 5', fontsize=16, fontweight='bold')

# 1. 主函数图
ax1 = plt.subplot(2, 3, 1)
x_vals = np.linspace(X_MIN, X_MAX, 400)
y_vals = f(x_vals)

# 绘制函数曲线
ax1.plot(x_vals, y_vals, 'b-', linewidth=3, label='y = -x² + 4x + 5', alpha=0.8)

# 标记极大值点
ax1.plot(2, 9, 'ro', markersize=12, label='极大值点 (2, 9)', zorder=5)

# 添加辅助线
ax1.axhline(y=9, color='r', linestyle='--', alpha=0.5, linewidth=1)
ax1.axvline(x=2, color='r', linestyle='--', alpha=0.5, linewidth=1)

# 填充极值区域
ax1.fill_between(x_vals, y_vals, 9, where=(y_vals <= 9), color='red', alpha=0.1)

# 设置图形属性
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('二次函数曲线及极大值点', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')
ax1.set_xlim(X_MIN, X_MAX)
ax1.set_ylim(Y_MIN, Y_MAX)

# 2. 一阶导数图
ax2 = plt.subplot(2, 3, 2)
y_prime_vals = f_prime(x_vals)

ax2.plot(x_vals, y_prime_vals, 'g-', linewidth=3, label="y' = -2x + 4", alpha=0.8)
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
ax2.axvline(x=2, color='r', linestyle='--', alpha=0.5, linewidth=1)

# 标记导数为0的点
ax2.plot(2, 0, 'ro', markersize=10, label="驻点 (2, 0)", zorder=5)

# 用颜色区分正负导数区域
ax2.fill_between(x_vals, 0, y_prime_vals, where=(y_prime_vals >= 0), color='green', alpha=0.2)
ax2.fill_between(x_vals, 0, y_prime_vals, where=(y_prime_vals < 0), color='red', alpha=0.2)

ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel("y'", fontsize=12)
ax2.set_title('一阶导数曲线', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')
ax2.set_xlim(X_MIN, X_MAX)
ax2.set_ylim(-10, 10)


# 3. 二阶导数图
ax3 = plt.subplot(2, 3, 3)
# 生成与x_vals同长度的常量数组
y_double_prime_vals = np.full_like(x_vals, f_double_prime(0))

ax3.plot(x_vals, y_double_prime_vals, 'r-', linewidth=3, label="y'' = -2", alpha=0.8)
ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=1)

# 标记二阶导数为负
ax3.fill_between(x_vals, 0, y_double_prime_vals, where=(y_double_prime_vals < 0), color='red', alpha=0.3)

ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel("y''", fontsize=12)
ax3.set_title('二阶导数（恒为负）', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right')
ax3.set_xlim(X_MIN, X_MAX)
ax3.set_ylim(-3, 1)

# 4. 极值分析图
ax4 = plt.subplot(2, 3, 4)

# 绘制函数曲线
ax4.plot(x_vals, y_vals, 'b-', linewidth=2, alpha=0.7)

# 标记关键点
critical_x = 2
critical_y = f(critical_x)

# 绘制切线
tangent_x = np.array([critical_x - 2, critical_x + 2])
tangent_y = f_prime(critical_x) * (tangent_x - critical_x) + critical_y
ax4.plot(tangent_x, tangent_y, 'g--', linewidth=2, label=f'切线 (斜率=0)', alpha=0.7)

# 标记极值点
ax4.plot(critical_x, critical_y, 'ro', markersize=12, label=f'极大值点 ({critical_x}, {critical_y})', zorder=5)

# 添加标注框
analysis_text = (
    "极值分析：\n"
    "1. 令 y' = -2x + 4 = 0\n"
    "   解得 x = 2\n"
    "2. y'' = -2 < 0\n"
    "   二阶导数为负\n"
    "3. 因此 (2, 9) 是极大值点"
)

ax4.text(0.05, 0.95, analysis_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax4.set_xlabel('x', fontsize=12)
ax4.set_ylabel('y', fontsize=12)
ax4.set_title('极值分析', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(loc='upper right')
ax4.set_xlim(-3, 7)
ax4.set_ylim(-10, 12)

# 5. 数值优化过程模拟
ax5 = plt.subplot(2, 3, 5)

# 模拟梯度下降过程
def gradient_descent(initial_x, learning_rate=0.1, iterations=15):
    """模拟梯度下降法寻找最大值"""
    x_history = [initial_x]
    y_history = [f(initial_x)]
    
    current_x = initial_x
    for i in range(iterations):
        # 计算梯度（沿梯度方向寻找最大值）
        gradient = f_prime(current_x)  # 一阶导数 = -2x + 4
        # 更新位置：沿梯度方向移动
        current_x = current_x + learning_rate * gradient
        x_history.append(current_x)
        y_history.append(f(current_x))
    
    return x_history, y_history

# 运行梯度下降从不同起点
initial_points = [-2, 0, 4, 6]
colors = ['green', 'blue', 'orange', 'purple']

# 绘制函数曲线
ax5.plot(x_vals, y_vals, 'b-', linewidth=2, alpha=0.5, label='y = -x² + 4x + 5')

# 绘制不同起点的优化路径
for init_point, color in zip(initial_points, colors):
    x_path, y_path = gradient_descent(init_point, learning_rate=0.2, iterations=10)
    ax5.plot(x_path, y_path, 'o-', color=color, linewidth=1.5, markersize=4,
            label=f'起点 x={init_point}')
    ax5.plot(init_point, f(init_point), 'o', color=color, markersize=8, markeredgecolor='black')

# 标记最终点
ax5.plot(2, 9, 'r*', markersize=20, label='收敛点 (2, 9)', zorder=10)

ax5.set_xlabel('x', fontsize=12)
ax5.set_ylabel('y', fontsize=12)
ax5.set_title('数值优化过程模拟', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend(loc='upper right', fontsize=9)
ax5.set_xlim(X_MIN, X_MAX)
ax5.set_ylim(Y_MIN, Y_MAX)

# 6. 结果总结图
ax6 = plt.subplot(2, 3, 6)

# 创建表格数据
methods = ['解析法', 'minimize', 'minimize_scalar', 'fminbound']
x_values = [2.0000000000, x_max1, x_max2, x_max3]
y_values = [9.0000000000, y_max1, y_max2, y_max3]
errors = [0.0000000000, abs(x_max1-2), abs(x_max2-2), abs(x_max3-2)]

# 隐藏坐标轴
ax6.axis('tight')
ax6.axis('off')

# 创建表格
table_data = []
for i in range(len(methods)):
    table_data.append([
        methods[i],
        f'{x_values[i]:.10f}',
        f'{y_values[i]:.10f}',
        f'{errors[i]:.10f}'
    ])

# 绘制表格
table = ax6.table(cellText=table_data,
                  colLabels=['方法', 'x值', 'y值', '误差'],
                  colWidths=[0.2, 0.3, 0.3, 0.2],
                  cellLoc='center',
                  loc='center')

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# 高亮解析法行
for i in range(4):
    if i == 0:  # 解析法行
        table[(i+1, 0)].set_facecolor('#FFFFCC')
        table[(i+1, 1)].set_facecolor('#FFFFCC')
        table[(i+1, 2)].set_facecolor('#FFFFCC')
        table[(i+1, 3)].set_facecolor('#FFFFCC')

ax6.set_title('结果对比表', fontsize=14, fontweight='bold', pad=20)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.92)

# 保存图像
plt.savefig('quadratic_maximum_analysis.png', dpi=FIGURE_DPI, bbox_inches='tight')
print("可视化图表已保存为 'quadratic_maximum_analysis.png'")

# 显示图像
plt.show()

# ==================== 第四部分：扩展分析 ====================
print("\n" + "=" * 60)
print("第四部分：扩展分析")
print("=" * 60)

print("\n1. 函数性质分析：")
print(f"   函数开口方向: 向下 (因为二次项系数为负)")
print(f"   对称轴: x = 2")
print(f"   顶点坐标: (2, 9)")
print(f"   与y轴交点: (0, 5)")
print(f"   与x轴交点: 解方程 -x² + 4x + 5 = 0")

# 计算与x轴交点
discriminant = 4**2 - 4*(-1)*5
if discriminant >= 0:
    x1 = (-4 + np.sqrt(discriminant)) / (2*(-1))
    x2 = (-4 - np.sqrt(discriminant)) / (2*(-1))
    print(f"               x1 = {x1:.2f}, x2 = {x2:.2f}")

print("\n2. 极值点验证：")
print("   在 x = 1.9 处: y =", f(1.9))
print("   在 x = 2.0 处: y =", f(2.0))
print("   在 x = 2.1 处: y =", f(2.1))
print("   可见 x=2 处确实为极大值点")

print("\n3. 优化算法对比：")
print("   - minimize: 通用优化算法，适用于多种问题")
print("   - minimize_scalar: 专门用于单变量函数优化")
print("   - fminbound: 有界区间内的最小值查找")
print("   对于二次函数，所有方法都能准确找到极值点")

print("\n4. 实际应用：")
print("   二次函数极值问题常见于：")
print("   - 物理学中的抛体运动")
print("   - 经济学中的成本收益分析")
print("   - 工程学中的最优化设计")

print("=" * 60)
print("\n程序执行完成！")