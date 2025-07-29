import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- 全局配置与调试开关 ---

# 1. 数据路径配置
# 训练数据目录 (使用动态阈值处理后的数据)
TRAIN_DATA_DIR = Path(r"/home/hello/quant/labeled_train")
# 官方测试集目录 (假设也已经过同样的预处理)
# !! 重要: 请确保测试集也经过了与训练集完全相同的特征工程和动态标签处理 !!
OFFICIAL_TEST_DIR = Path(r"/home/hello/quant/labeled_test") 

# 2. 调试与抽样配置
# 为了快速调试，可以只抽样部分文件。设置为 None 将分析所有文件。
# 例如, SAMPLE_TRAIN_FILES = 100 只随机读取100个训练文件。
SAMPLE_TRAIN_FILES = 200 
SAMPLE_TEST_FILES = None # 加载所有官方测试文件

# 3. 模型与训练参数
LGBM_PARAMS = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss', # 对数损失是分类任务的常用评估指标
    'boosting_type': 'gbdt',
    'device': 'gpu',  # !! 核心：启用GPU加速 !!
    'gpu_platform_id': 0, # 通常为0
    'gpu_device_id': 0,   # 通常为0
    
    # --- 以下为性能和正则化调优参数 ---
    'n_estimators': 2000, # 初始设置一个较大的树的数量，通过早停来找到最佳值
    'learning_rate': 0.02,
    'num_leaves': 63, # 建议值: 2^(max_depth) - 1
    'max_depth': 6,
    'seed': 42,
    'n_jobs': -1, # 使用所有可用的CPU核心进行数据加载和预处理
    'verbose': -1, # 关闭LightGBM自身的啰嗦日志
    'colsample_bytree': 0.8, # 特征采样
    'subsample': 0.8, # 数据采样
}

# 早停参数：如果在100轮内验证集性能没有提升，则停止训练
EARLY_STOPPING_ROUNDS = 100
# 验证集划分比例：从训练数据中划分出后20%作为“未来时段”验证集
VALIDATION_SET_RATIO = 0.2

# --- Matplotlib 全局美化设置 ---
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 7)


def load_data(input_dir: Path, sample_size: int = None, description="加载数据中") -> pd.DataFrame:
    """从指定目录加载多个Parquet文件，支持抽样。"""
    files = list(input_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"在目录 {input_dir} 中未找到任何 .parquet 文件。")
    
    if sample_size and sample_size < len(files):
        print(f"为 '{description}' 随机抽样 {sample_size} 个文件...")
        np.random.seed(42)
        files = np.random.choice(files, sample_size, replace=False)

    all_dfs = [pd.read_parquet(file) for file in tqdm(files, desc=description)]
    
    if not all_dfs:
        raise ValueError(f"未能从 {input_dir} 成功加载任何数据文件。")
        
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['tradingtime'] = pd.to_datetime(full_df['tradingtime'])
    return full_df.sort_values('tradingtime').reset_index(drop=True)

def plot_learning_curves(evals_result):
    """绘制学习曲线。"""
    plt.figure()
    plt.plot(evals_result['training']['multi_logloss'], label='训练集 LogLoss')
    plt.plot(evals_result['validation']['multi_logloss'], label='验证集 LogLoss')
    plt.title('模型训练过程中的学习曲线')
    plt.xlabel('Boosting 轮次')
    plt.ylabel('Multi-LogLoss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title='混淆矩阵'):
    """绘制混淆矩阵。"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[-1, 0, 1], yticklabels=[-1, 0, 1])
    plt.title(title)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()

def main():
    """主函数：加载数据、训练模型、评估并可视化。"""
    # --- 1. 数据加载与准备 ---
    print("--- 开始加载数据 ---")
    # 加载主数据集（用于训练和未来时段验证）
    main_df = load_data(TRAIN_DATA_DIR, sample_size=SAMPLE_TRAIN_FILES, description="加载训练/验证数据")
    
    # 加载官方测试集
    try:
        official_test_df = load_data(OFFICIAL_TEST_DIR, sample_size=SAMPLE_TEST_FILES, description="加载官方测试数据")
    except FileNotFoundError as e:
        print(f"警告: {e}\n将跳过对官方测试集的评估。")
        official_test_df = None

    # --- 2. 数据集划分 ---
    print("\n--- 准备特征 (X) 和标签 (Y) ---")
    
    # 定义特征列和标签列
    label_col = 'label'
    # 自动识别所有非ID和非标签的列作为特征
    features = [col for col in main_df.columns if col not in ['tradingtime', 'symbol', 'label']]
    
    # 确保官方测试集也有同样的列
    if official_test_df is not None:
        features = [f for f in features if f in official_test_df.columns]
        official_test_df = official_test_df[features + [label_col]]
        
    main_df = main_df[features + [label_col]].dropna()

    # 标签转换: LightGBM需要从0开始的整数标签 (0, 1, 2)
    main_df[label_col] = main_df[label_col] + 1
    if official_test_df is not None:
        official_test_df[label_col] = official_test_df[label_col] + 1

    # 按时间划分训练集和“未来时段”验证集
    val_size = int(len(main_df) * VALIDATION_SET_RATIO)
    train_df = main_df.iloc[:-val_size]
    val_df = main_df.iloc[-val_size:]

    X_train, y_train = train_df[features], train_df[label_col]
    X_val, y_val = val_df[features], val_df[label_col]

    print(f"训练集大小: {len(X_train)} | 验证集大小: {len(X_val)}")
    if official_test_df is not None:
        X_test_official, y_test_official = official_test_df[features], official_test_df[label_col]
        print(f"官方测试集大小: {len(X_test_official)}")

    # 创建LightGBM数据集
    lgb_train = lgb.Dataset(X_train, y_train, feature_name=features)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, feature_name=features)

    # --- 3. 模型训练 ---
    print("\n--- 开始使用GPU进行模型训练 ---")
    evals_result = {}  # 用于存储评估结果以绘制学习曲线
    
    model = lgb.train(
        params=LGBM_PARAMS,
        train_set=lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['training', 'validation'],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=10), # 每10轮打印一次日志
            lgb.record_evaluation(evals_result) # !! 核心改动：使用回调函数记录评估结果 !!
        ]
    )
    
    # 保存模型
    model.save_model('lgbm_model.txt', num_iteration=model.best_iteration)
    print(f"\n模型训练完成！最佳迭代轮次: {model.best_iteration}")

    # --- 4. 结果评估与可视化 ---
    print("\n--- 正在评估模型性能 ---")
    
    # 绘制学习曲线
    plot_learning_curves(evals_result)
    
    # 在“未来时段”验证集上评估
    print("\n--- 1. 在'未来时段'验证集上的表现 ---")
    y_pred_val_proba = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred_val = np.argmax(y_pred_val_proba, axis=1)
    
    print("分类报告:")
    # 将标签转换回原始的-1, 0, 1以便阅读
    print(classification_report(y_val - 1, y_pred_val - 1))
    plot_confusion_matrix(y_val - 1, y_pred_val - 1, title='未来时段验证集 - 混淆矩阵')

    # 在官方测试集上评估
    if official_test_df is not None:
        print("\n--- 2. 在'官方测试集'上的表现 ---")
        y_pred_test_proba = model.predict(X_test_official, num_iteration=model.best_iteration)
        y_pred_test = np.argmax(y_pred_test_proba, axis=1)
        
        print("分类报告:")
        print(classification_report(y_test_official - 1, y_pred_test - 1))
        plot_confusion_matrix(y_test_official - 1, y_pred_test - 1, title='官方测试集 - 混淆矩阵')


if __name__ == "__main__":
    # !! 重要提示: 运行此脚本前，请确保您已安装支持GPU的LightGBM版本。
    # 安装命令通常为: pip install lightgbm --install-option=--gpu
    # 您还需要正确安装NVIDIA驱动和CUDA Toolkit。
    main()
