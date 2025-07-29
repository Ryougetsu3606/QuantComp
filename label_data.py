# ==============================================================================
# 文件: label_data.py (优化版-仅打标)
# 描述: 本脚本读取已完成特征工程的数据，并基于“先过滤，后打标”的逻辑生成标签。
#
# **工作流程**:
# 1. 读取已包含 'WAP', 'OBI_L1', 'Volatility' 等特征的文件。
# 2. 增加事件过滤器: 只有当市场的订单簿不平衡(OBI_L1)超过一定阈值时，
#    才认为这是一个“值得关注”的事件。
# 3. 条件化打标: 只对被过滤器识别出的事件时间点，应用三道门方法进行打标。
# 4. 保存最终包含标签的数据，用于模型训练。
# ==============================================================================

import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import numba

# --- 核心打标逻辑 ---

@numba.jit(nopython=True)
def get_labels_for_events(prices, volatility, event_indices, pt_multiplier, sl_multiplier, time_limit):
    """
    仅为被触发的事件点位构建三道门标签 (Numba加速版)。
    """
    n_points = len(prices)
    labels = np.zeros(n_points, dtype=np.int8) # 标签默认为0

    for i in event_indices:
        # 确保不会索引越界
        if i >= n_points - time_limit:
            continue

        current_price = prices[i]
        vol = volatility[i]
        
        if np.isnan(vol) or vol == 0:
            continue
            
        upper_barrier = current_price * (1 + vol * pt_multiplier)
        lower_barrier = current_price * (1 - vol * sl_multiplier)
        
        for j in range(1, time_limit + 1):
            future_price = prices[i + j]
            if future_price >= upper_barrier:
                labels[i] = 1 # 命中上轨
                break
            elif future_price <= lower_barrier:
                labels[i] = -1 # 命中下轨
                break
    
    return labels

def get_event_triggers(obi_series, window, threshold):
    """
    根据OBI的变化来确定事件触发点。
    当OBI的滚动均值超过阈值时，触发事件。
    """
    # 计算OBI的滚动均值，取绝对值表示不平衡的程度
    obi_rolling_mean = obi_series.abs().rolling(window=window, min_periods=window).mean()
    
    # 当滚动均值超过阈值时，我们认为是一个事件
    triggers = obi_rolling_mean > threshold
    
    # 为避免信号过于密集，可以加入一个冷却期逻辑
    # 这里简化处理：一旦触发，短时间内不再重复触发
    # (更复杂的逻辑可以确保两次触发之间有最小间隔)
    
    return triggers


# --- 文件处理主函数 ---

def process_and_label_file(input_file, output_file, event_window, event_threshold, pt_multiplier, sl_multiplier, time_limit):
    """
    对单个已完成特征工程的文件进行标签构建。
    """
    df = pd.read_parquet(input_file)

    # 1. 【已修改】检查打标所需的特征列是否存在
    required_features = ['WAP', 'OBI_L1', 'Volatility']
    if not all(col in df.columns for col in required_features):
        print(f"错误: 文件 {input_file.name} 缺少必要的特征列。需要 {required_features}。请先运行特征工程脚本。")
        return

    # 2. 【已修改】使用 OBI_L1 确定事件触发点
    event_triggers = get_event_triggers(df['OBI_L1'], window=event_window, threshold=event_threshold)
    event_indices = np.where(event_triggers)[0] # 获取触发点的索引

    # 3. 为事件点打标
    labels = get_labels_for_events(
        prices=df['WAP'].to_numpy(dtype=np.float64),
        volatility=df['Volatility'].to_numpy(dtype=np.float64),
        event_indices=event_indices,
        pt_multiplier=pt_multiplier,
        sl_multiplier=sl_multiplier,
        time_limit=time_limit
    )
    df['label'] = labels

    # 4. 清理并保存数据
    # 删除包含NaN的行 (通常是窗口计算的初始部分)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 直接在原数据上增加label列并保存
    df.to_parquet(output_file, index=False)


def batch_process_files(input_root_dir, output_root_dir, **kwargs):
    """
    批量处理目录下的所有文件。
    """
    input_path = Path(input_root_dir)
    output_path = Path(output_root_dir)

    files_to_process = list(input_path.glob('**/*.parquet'))
    if not files_to_process:
        print(f"在目录 {input_root_dir} 中没有找到任何 .parquet 文件。")
        return

    print(f"共找到 {len(files_to_process)} 个文件需要处理。")

    for file in tqdm(files_to_process, desc="批量处理进度"):
        relative_path = file.relative_to(input_path)
        output_file_path = output_path / relative_path
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        process_and_label_file(file, output_file_path, **kwargs)

    print("批量处理与标签构建完成！")


if __name__ == '__main__':
    # --- 参数配置 ---
    
    # 事件过滤窗口 (用于计算OBI均值)
    EVENT_WINDOW = 100 # 约50秒
    
    # 事件触发阈值 (OBI滚动均值的绝对值)
    # 这是一个关键参数，需要反复试验调整
    EVENT_THRESHOLD = 0.4 
    
    # 盈利阈值乘数 (可以适当放大)
    PROFIT_TAKE_MULTIPLIER = 3.0
    
    # 止损阈值乘数 (可以适当放大)
    STOP_LOSS_MULTIPLIER = 3.0
    
    # 最长持有期 (单位: ticks)
    TIME_LIMIT_TICKS = 40 # 20秒
    
    # --- 路径配置 ---
    
    # 【请修改】输入目录：存放您已经过特征工程处理的数据 (例如 deal_data.ipynb 的输出)
    INPUT_DATA_ROOT = "/home/hello/quant/processed_test"
    
    # 【请修改】输出目录：存放经过本脚本打标后的、可用于训练的最终数据
    OUTPUT_DATA_ROOT = "/home/hello/quant/labeled_test_filtered"

    # --- 开始执行 ---
    batch_process_files(
        input_root_dir=INPUT_DATA_ROOT,
        output_root_dir=OUTPUT_DATA_ROOT,
        event_window=EVENT_WINDOW,
        event_threshold=EVENT_THRESHOLD,
        pt_multiplier=PROFIT_TAKE_MULTIPLIER,
        sl_multiplier=STOP_LOSS_MULTIPLIER,
        time_limit=TIME_LIMIT_TICKS
    )
