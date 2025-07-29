# ==============================================================================
# File: generate_submission.py
# Description:
#   This script generates the final position files for submission. It loads the
#   best pre-trained model, runs predictions on the test set, and saves the
#   positions in the format required by the competition.
#
# How to Run:
#   1. Ensure you have a saved model from the training script
#      (e.g., in './lgbm_models/lgbm_best_sharpe_model.txt').
#   2. Make sure the paths in the configuration section are correct.
#   3. Run this script from your terminal:
#      $ python generate_submission.py
#   4. The position files will be created in the specified submission directory.
# ==============================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
import shutil
from pathlib import Path
from tqdm import tqdm

# --- Configuration Section ---

# 1. PATHS
# Path to the directory containing your preprocessed test data
TEST_DATA_DIR = Path("/home/hello/quant/labeled_test")
# Path to the saved model you want to use for prediction
MODEL_PATH = Path("/home/hello/lgbm_models/lgbm_best_accuracy_model400.txt")
# The root directory for your submission files
SUBMISSION_DIR = Path("./final_submit/positions")

# 2. FEATURES
# Must be the same list used for training
FEATURES = [
    'WAP_norm', 'OBI_L1_norm', 'OBI_L2_norm', 'Spread_L1_norm',
    'DepthImbalance_norm', 'Momentum_norm', 'Volatility', 'TimeOfDay'
]

# --- Main Script ---

def generate_submission_files():
    """Loads a model, runs predictions on all test files, and saves them in the submission format."""
    
    # 1. Clean and create the submission directory
    if SUBMISSION_DIR.exists():
        print(f"正在清理已存在的提交目录: {SUBMISSION_DIR}")
        shutil.rmtree(SUBMISSION_DIR)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    print(f"已创建空的提交目录: {SUBMISSION_DIR}")

    # 2. Load the Model
    if not MODEL_PATH.exists():
        print(f"错误: 在路径 {MODEL_PATH} 未找到模型文件。请先运行训练脚本。")
        return
    print(f"正在从 {MODEL_PATH} 加载模型...")
    model = lgb.Booster(model_file=str(MODEL_PATH))
    print("模型加载成功。")

    # 3. Load and process each test file individually
    all_test_files = list(TEST_DATA_DIR.glob("**/*.parquet"))
    if not all_test_files:
        print(f"错误: 在目录 {TEST_DATA_DIR} 中未找到测试数据文件。")
        return
    
    print(f"找到 {len(all_test_files)} 个测试文件需要处理...")
    for file_path in tqdm(all_test_files, desc="正在生成提交文件"):
        try:
            df_test = pd.read_parquet(file_path)
            if df_test.empty:
                continue

            # Ensure necessary columns exist before prediction
            if not all(f in df_test.columns for f in FEATURES) or 'SYMBOL' not in df_test.columns:
                print(f"警告: 文件 {file_path.name} 缺少必要的特征或SYMBOL列，已跳过。")
                continue

            X_test = df_test[FEATURES].astype(np.float32)

            # Generate predictions by taking the class with the highest probability
            predicted_probs = model.predict(X_test)
            predicted_labels_mapped = np.argmax(predicted_probs, axis=1)
            
            label_map_reverse = {0: -1, 1: 0, 2: 1}
            df_test['position'] = pd.Series(predicted_labels_mapped).map(label_map_reverse).values

            # Prepare the final submission DataFrame
            submission_df = df_test[['TRADINGTIME', 'position']].copy()

            # Get date and symbol to construct the output path
            date_str = df_test['TRADINGTIME'].iloc[0].strftime('%Y%m%d')
            # Extract the contract name from the original filename
            contract_name = file_path.stem
            
            # Create the date-specific subdirectory
            output_date_dir = SUBMISSION_DIR / date_str
            output_date_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the final CSV file
            output_file_path = output_date_dir / f"{contract_name}.csv"
            submission_df.to_csv(output_file_path, index=False)

        except Exception as e:
            print(f"处理文件 {file_path.name} 时发生错误: {e}")

    print("\n--- 所有提交文件已生成完毕 ---")
    print(f"请在以下目录中检查您的文件: {SUBMISSION_DIR}")


if __name__ == "__main__":
    generate_submission_files()
