# ==============================================
# 20辆电动车充电负荷优化分析完整代码（Pylance兼容修复版）
# 功能：负荷计算可视化 + 波动率量化 + 优化建模求解 + 结果对比
# ==============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
import os
import argparse
from typing import Tuple, Dict, Any

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SAVE_PATH = os.path.abspath('.')  # 默认当前目录
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH, exist_ok=True)

def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, list, pd.DatetimeIndex, float]:
    """
    加载双工作表数据（Initialization + Activity）并预处理
    返回：功率矩阵DataFrame、ev列名列表、时间索引、时间粒度（小时）
    """
    try:
        df_init = pd.read_excel(file_path, sheet_name='Initialization', engine='openpyxl')
        df_activity = pd.read_excel(file_path, sheet_name='Activity', engine='openpyxl')
        print(f"✅ 成功加载双工作表数据：{file_path}")
        print(f"- Initialization表形状：{df_init.shape}")
        print(f"- Activity表形状：{df_activity.shape}")
    except Exception as e:
        print(f"⚠️ 无法以Excel读取，尝试以CSV降级：{e}")
        base = os.path.splitext(os.path.basename(file_path))[0]
        try:
            df_init = pd.read_csv(f"{base} - Initialization.csv")
            df_activity = pd.read_csv(f"{base} - Activity.csv")
            print("✅ 已降级读取CSV工作表文件")
        except Exception as e2:
            raise RuntimeError(f"数据加载失败（Excel/CSV均不可读）：{e2}")

    num_evs = 20
    time_step = 0.25  # 小时（15分钟）
    time_slots = int(24 / time_step)
    time_interval = time_step

    time_index = pd.date_range(start='2026-01-01 00:00', periods=time_slots, freq='15min')

    P_baseline = np.zeros((num_evs, time_slots))
    Availability = np.zeros((num_evs, time_slots))
    P_max_vec = np.zeros(num_evs)

    for vid in range(1, num_evs + 1):
        car_act = df_activity[df_activity['Vehicle ID'] == vid]
        if len(car_act) > 0 and 'P_max (W)' in car_act.columns:
            p_max_val = car_act['P_max (W)'].iloc[0]
            P_max_vec[vid - 1] = p_max_val if p_max_val > 0 else 1440
        else:
            P_max_vec[vid - 1] = 1440

    for vid in range(1, num_evs + 1):
        car_act = df_activity[df_activity['Vehicle ID'] == vid]
        for _, row in car_act.iterrows():
            try:
                start_idx = int(row['Start time (hour)'] / time_step)
                end_idx = int(row['End time (hour)'] / time_step)
            except Exception:
                continue
            end_idx = min(end_idx, time_slots)
            if 'State' in row and row['State'] == 'Charging':
                P_baseline[vid - 1, start_idx:end_idx] = P_max_vec[vid - 1]
            if 'Location' in row and row['Location'] == 'Home' and ('State' not in row or row['State'] != 'Driving'):
                Availability[vid - 1, start_idx:end_idx] = 1

    P_baseline_kW = P_baseline / 1000.0
    P_max_vec_kW = P_max_vec / 1000.0

    ev_power_cols = [f'EV{vid}' for vid in range(1, num_evs + 1)]
    df_power = pd.DataFrame(P_baseline_kW.T, columns=ev_power_cols)
    df_power['时间'] = time_index

    global charging_availability, ev_max_power
    charging_availability = Availability.T  # (time, vehicle)
    ev_max_power = P_max_vec_kW

    return df_power, ev_power_cols, time_index, time_interval

def calculate_and_plot_load(
    df: pd.DataFrame,
    ev_power_cols: list,
    time_index: pd.DatetimeIndex,
    time_interval: float,
    save_path: str
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df['总充电负荷'] = df[ev_power_cols].sum(axis=1)

    load_stats = {
        '峰值负荷_kW': df['总充电负荷'].max(),
        '谷值负荷_kW': df['总充电负荷'].min(),
        '平均负荷_kW': df['总充电负荷'].mean(),
        '总充电量_kWh': df['总充电负荷'].sum() * time_interval,
        '数据点数量': len(df)
    }

    print("\n=== 原始充电负荷统计指标 ===")
    for k, v in load_stats.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")

    plt.figure(figsize=(14, 8))
    plt.plot(time_index, df['总充电负荷'], color='#2E86AB', linewidth=2.5, label='总充电负荷')
    plt.axhline(y=load_stats['平均负荷_kW'], color='#A23B72', linestyle='--', linewidth=2,
                label=f'平均负荷: {load_stats["平均负荷_kW"]:.2f}kW')
    plt.axhline(y=load_stats['峰值负荷_kW'], color='#F18F01', linestyle=':', linewidth=2,
                label=f'峰值负荷: {load_stats["峰值负荷_kW"]:.2f}kW')

    plt.title('20辆电动车原始充电负荷曲线（15分钟粒度）', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('充电负荷 (kW)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'original_charging_load_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    save_cols = ['时间'] + ev_power_cols + ['总充电负荷']
    df[save_cols].to_csv(os.path.join(save_path, 'original_charging_load_data.csv'),
                         index=False, encoding='utf-8-sig')

    print(f"\n✅ 原始负荷曲线已保存至：{os.path.join(save_path, 'original_charging_load_curve.png')}")
    return df, load_stats

def calculate_volatility(load_series: pd.Series, time_interval: float) -> Dict[str, float]:
    load = load_series.values
    load_mean = np.mean(load) if len(load) > 0 else 0.0
    load_std = np.std(load)
    load_max = np.max(load) if len(load) > 0 else 0.0
    load_min = np.min(load) if len(load) > 0 else 0.0

    # 处理变化率除零风险：用nan-safe计算然后填0
    diffs = np.diff(load)
    denom = load[:-1].copy()
    denom_safe = np.where(denom == 0, np.nan, denom)
    change_rates = np.abs(diffs) / denom_safe
    max_change_rate = np.nanmax(change_rates) if np.any(~np.isnan(change_rates)) else 0.0
    mean_change_rate = np.nanmean(change_rates) if np.any(~np.isnan(change_rates)) else 0.0

    indicators = {
        '负荷方差_kW2': np.var(load),
        '负荷标准差_kW': load_std,
        '变异系数': load_std / load_mean if load_mean != 0 else 0,
        '峰谷差_kW': load_max - load_min,
        '峰谷差率': (load_max - load_min) / load_min if load_min != 0 else 0,
        '最大负荷变化率': max_change_rate,
        '平均负荷变化率': mean_change_rate,
        '高负荷占比': np.sum(load >= load_mean * 1.2) / len(load) if len(load) > 0 else 0,
        '低负荷占比': np.sum(load <= load_mean * 0.8) / len(load) if len(load) > 0 else 0,
        '累积绝对波动_kWh': np.sum(np.abs(diffs)) * time_interval,
        '平均负荷_kW': load_mean
    }

    return indicators

def plot_volatility_analysis(load_series: pd.Series, volatility: Dict[str, float], save_path: str) -> pd.DataFrame:
    vol_df = pd.DataFrame({
        '指标名称': list(volatility.keys()),
        '数值': list(volatility.values())
    })

    print("\n=== 原始充电负荷波动率指标 ===")
    print(vol_df.to_string(index=False))

    plt.figure(figsize=(16, 10))
    ax1 = plt.subplot(2, 1, 1)
    indicators = vol_df['指标名称'][:9]
    values = vol_df['数值'][:9]
    colors = plt.cm.Set3(np.linspace(0, 1, len(indicators)))

    bars = ax1.bar(indicators, values, color=colors, alpha=0.8)
    ax1.set_title('原始充电负荷波动率指标分析', fontsize=14, fontweight='bold')
    ax1.set_ylabel('指标数值', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)

    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    ax2 = plt.subplot(2, 1, 2)
    ax2.hist(load_series, bins=20, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax2.axvline(load_series.mean(), color='#A23B72', linestyle='--', linewidth=2,
                label=f'均值: {load_series.mean():.2f}kW')
    ax2.set_title('原始充电负荷分布直方图', fontsize=14, fontweight='bold')
    ax2.set_xlabel('充电负荷 (kW)', fontsize=12)
    ax2.set_ylabel('频次', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'charging_load_volatility_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    vol_df.to_csv(os.path.join(save_path, 'charging_load_volatility_results.csv'),
                  index=False, encoding='utf-8-sig')

    print(f"\n✅ 波动率分析图表已保存至：{os.path.join(save_path, 'charging_load_volatility_analysis.png')}")
    return vol_df

def optimize_charging_load(
    df: pd.DataFrame,
    ev_power_cols: list,
    time_index: pd.DatetimeIndex,
    time_interval: float,
    save_path: str
) -> Tuple[pd.DataFrame, np.ndarray]:
    n_ev = len(ev_power_cols)
    n_time = len(df)

    ev_energy_demand = (df[ev_power_cols].values.T * time_interval).sum(axis=1)
    global ev_max_power
    global charging_availability

    print(f"\n=== 优化问题维度 ===")
    print(f"- 车辆数量：{n_ev}辆")
    print(f"- 时间点数量：{n_time}个")
    print(f"- 优化变量数量：{n_ev * n_time}个")

    def objective(x: np.ndarray) -> float:
        power_matrix = x.reshape(n_ev, n_time)
        total_load = np.sum(power_matrix, axis=0)
        return np.var(total_load)

    def constraint_energy(x: np.ndarray) -> np.ndarray:
        power_matrix = x.reshape(n_ev, n_time)
        energy_supplied = np.sum(power_matrix * time_interval, axis=1)
        return energy_supplied - ev_energy_demand

    x0 = df[ev_power_cols].values.T.flatten()

    # 优化的bounds设置：直接整合功率和可用性约束
    bounds = []
    for ev_idx in range(n_ev):
        max_p = float(ev_max_power[ev_idx])
        for time_idx in range(n_time):
            # 如果该时段不可用，强制为0；否则在[0, max_power]范围内
            if charging_availability[time_idx, ev_idx] == 1:
                bounds.append((0.0, max_p))
            else:
                bounds.append((0.0, 0.0))

    # 仅保留必要的等式约束（能量约束）
    constraints = [
        {'type': 'eq', 'fun': constraint_energy}
    ]

    print("\n=== 开始优化求解（已优化算法参数）===")
    result = minimize(
        fun=objective,
        x0=x0,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP',
        options={'maxiter': 200, 'ftol': 1e-4, 'maxfev': 1000, 'disp': False}
    )

    if not result.success:
        print(f"⚠️ 优化注意：{result.message}，使用当前结果继续")

    print(f"✅ 优化完成！迭代次数：{getattr(result, 'nit', 'N/A')}，最终方差：{result.fun:.6f}")

    optimized_power = result.x.reshape(n_ev, n_time)
    optimized_total_load = np.sum(optimized_power, axis=0)

    optimized_df = pd.DataFrame()
    optimized_df['时间'] = time_index
    for i, col in enumerate(ev_power_cols):
        optimized_df[f'{col}_优化后'] = optimized_power[i, :]
    optimized_df['总充电负荷_优化后'] = optimized_total_load

    print("\n=== 约束验证 ===")
    optimized_energy = np.sum(optimized_power * time_interval, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.where(ev_energy_demand == 0, 1.0, ev_energy_demand)
        rel_errors = np.abs((optimized_energy - ev_energy_demand) / denom)
    energy_error = np.max(rel_errors) if rel_errors.size > 0 else 0.0
    print(f"电量需求误差：{energy_error * 100:.6f}%")

    power_violation = np.max(optimized_power - ev_max_power.reshape(-1, 1))
    print(f"最大功率超出量：{power_violation:.6f}kW")

    optimized_df.to_csv(os.path.join(save_path, 'optimized_charging_load_data.csv'),
                        index=False, encoding='utf-8-sig')

    schedule_df = pd.DataFrame(optimized_power.T, columns=[f'{col}_优化功率' for col in ev_power_cols])
    schedule_df.insert(0, '时间', time_index)
    schedule_df.to_csv(os.path.join(save_path, 'charging_schedule_optimized.csv'),
                       index=False, encoding='utf-8-sig')

    print(f"\n✅ 优化结果已保存至：{os.path.join(save_path, 'optimized_charging_load_data.csv')}")
    return optimized_df, optimized_total_load

def compare_optimization_results(
    df: pd.DataFrame,
    optimized_df: pd.DataFrame,
    optimized_total_load: np.ndarray,
    original_vol: Dict[str, float],
    time_index: pd.DatetimeIndex,
    time_interval: float,
    save_path: str
) -> None:
    optimized_vol = calculate_volatility(pd.Series(optimized_total_load), time_interval)

    comparison_df = pd.DataFrame({
        '指标名称': list(original_vol.keys()),
        '优化前数值': list(original_vol.values()),
        '优化后数值': list(optimized_vol.values()),
        '改善率_%': [
            round((1 - opt / orig) * 100, 2) if orig != 0 else 0
            for orig, opt in zip(original_vol.values(), optimized_vol.values())
        ]
    })

    print("\n=== 优化前后波动率对比 ===")
    print(comparison_df.to_string(index=False))

    plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(time_index, df['总充电负荷'], color='#2E86AB', linewidth=2.5,
             label=f'优化前（方差：{original_vol.get("负荷方差_kW2", 0):.2f}）')
    ax1.plot(time_index, optimized_total_load, color='#F18F01', linewidth=2.5,
             label=f'优化后（方差：{optimized_vol.get("负荷方差_kW2", 0):.2f}）')
    ax1.axhline(y=original_vol.get('平均负荷_kW', 0), color='#2E86AB', linestyle='--', alpha=0.7,
                label=f'优化前均值：{original_vol.get("平均负荷_kW", 0):.2f}kW')
    ax1.axhline(y=optimized_vol.get('平均负荷_kW', 0), color='#F18F01', linestyle='--', alpha=0.7,
                label=f'优化后均值：{optimized_vol.get("平均负荷_kW", 0):.2f}kW')
    ax1.set_title('充电负荷优化前后对比', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('充电负荷 (kW)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.tick_params(axis='x', rotation=45)

    ax2 = plt.subplot(2, 1, 2)
    key_inds = ['负荷方差_kW2', '变异系数', '峰谷差率', '最大负荷变化率']
    orig_vals = [original_vol.get(ind, 0) for ind in key_inds]
    opt_vals = [optimized_vol.get(ind, 0) for ind in key_inds]

    x = np.arange(len(key_inds))
    width = 0.35
    ax2.bar(x - width / 2, orig_vals, width, label='优化前', color='#2E86AB', alpha=0.8)
    ax2.bar(x + width / 2, opt_vals, width, label='优化后', color='#F18F01', alpha=0.8)
    ax2.set_title('关键波动率指标对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('指标数值', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(key_inds, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    for bar in ax2.patches:
        if bar.get_height() > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                     f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'charging_load_optimization_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    comparison_df.to_csv(os.path.join(save_path, 'volatility_comparison.csv'),
                         index=False, encoding='utf-8-sig')

    print(f"\n✅ 对比图表已保存至：{os.path.join(save_path, 'charging_load_optimization_comparison.png')}")

    print("\n=== 核心结论 ===")
    try:
        variance_improve = comparison_df[comparison_df['指标名称'] == '负荷方差_kW2']['改善率_%'].iloc[0]
        cv_improve = comparison_df[comparison_df['指标名称'] == '变异系数']['改善率_%'].iloc[0]
        pv_improve = comparison_df[comparison_df['指标名称'] == '峰谷差率']['改善率_%'].iloc[0]
        print(f"1. 负荷方差改善率：{variance_improve:.2f}%")
        print(f"2. 变异系数改善率：{cv_improve:.2f}%")
        print(f"3. 峰谷差率改善率：{pv_improve:.2f}%")
    except Exception:
        pass

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='20辆电动车充电负荷分析与优化')
    parser.add_argument('-i', '--input', help='输入Excel/CSV文件路径（优先）', default=None)
    parser.add_argument('-s', '--save', help='结果保存路径（默认当前目录）', default=SAVE_PATH)
    args = parser.parse_args(argv)

    input_file = None
    if args.input:
        if os.path.exists(args.input):
            input_file = args.input
        else:
            print(f"❌ 指定输入文件不存在：{args.input}")
            return

    if input_file is None:
        candidates = [f for f in os.listdir('.') if f.lower().endswith(('.xlsx', '.xls', '.csv'))]
        # 优先选择包含 'DrivingData' 的文件
        driving_files = [f for f in candidates if 'DrivingData' in f and f.lower().endswith(('.xlsx', '.xls'))]
        if driving_files:
            input_file = driving_files[0]
        elif 'DrivingData_20EVs.xlsx' in candidates:
            input_file = 'DrivingData_20EVs.xlsx'
        elif candidates:
            # 过滤掉输出结果文件
            data_candidates = [f for f in candidates if not any(x in f for x in ['volatility', 'optimized', 'schedule', 'comparison'])]
            if data_candidates:
                input_file = data_candidates[0]
            else:
                input_file = candidates[0]
        else:
            print("❌ 未找到可用的数据文件。请通过 -i 指定 DrivingData_20EVs.xlsx 或置于当前目录。")
            print("当前目录文件：", os.listdir('.'))
            return

    save_path = args.save
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    try:
        df, ev_cols, time_idx, time_int = load_and_preprocess_data(input_file)
        df, load_stats = calculate_and_plot_load(df, ev_cols, time_idx, time_int, save_path)
        original_vol = calculate_volatility(df['总充电负荷'], time_int)
        vol_df = plot_volatility_analysis(df['总充电负荷'], original_vol, save_path)
        optimized_df, optimized_load = optimize_charging_load(df, ev_cols, time_idx, time_int, save_path)
        compare_optimization_results(df, optimized_df, optimized_load,
                                     original_vol, time_idx, time_int, save_path)
        print("\n🎉 所有分析流程执行完成！")
    except Exception as e:
        print(f"\n❌ 程序执行出错：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])