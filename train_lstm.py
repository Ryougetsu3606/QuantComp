import pandas as pd
import numpy as np
from pathlib import Path

import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

# --- 全局配置 ---

# 1. 数据路径配置
TRAIN_DATA_DIR = Path(r"/home/hello/quant/labeled_train")
OFFICIAL_TEST_DIR = Path(r"/home/hello/quant/labeled_test") 

# 2. 调试与抽样配置
SAMPLE_TRAIN_FILES = None
SAMPLE_TEST_FILES = None

# 3. LSTM 与训练参数
SEQUENCE_LENGTH = 50
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SET_RATIO = 0.2

# --- Matplotlib 全局美化设置 ---
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 7)

# --- 数据加载与准备 ---

def load_data(input_dir: Path, sample_size: int = None, description="加载数据中") -> pd.DataFrame:
    """从指定目录加载多个Parquet文件。"""
    files = list(input_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"在目录 {input_dir} 中未找到任何 .parquet 文件。")
    
    if sample_size and sample_size < len(files):
        print(f"为 '{description}' 随机抽样 {sample_size} 个文件...")
        np.random.seed(42)
        files = np.random.choice(files, sample_size, replace=False)

    all_dfs = [pd.read_parquet(file) for file in files]
    if not all_dfs:
        raise ValueError(f"未能从 {input_dir} 成功加载任何数据文件。")
        
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['tradingtime'] = pd.to_datetime(full_df['tradingtime'])
    return full_df.sort_values('tradingtime').reset_index(drop=True)

class TimeSeriesDataset(Dataset):
    """自定义PyTorch数据集用于时间序列。"""
    def __init__(self, features, labels, sequence_length):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, index):
        return (self.features[index:index+self.sequence_length], 
                self.labels[index+self.sequence_length])

# --- 模型定义 ---

class LSTMModel(nn.Module):
    """基于PyTorch的LSTM模型架构。"""
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, 
            batch_first=True, dropout=dropout_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

# --- 绘图函数 ---

def plot_history(history):
    """绘制训练历史曲线。"""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练集 Loss')
    plt.plot(history['val_loss'], label='验证集 Loss')
    plt.title('模型 Loss 曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练集 Accuracy')
    plt.plot(history['val_acc'], label='验证集 Accuracy')
    plt.title('模型 Accuracy 曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_torch(y_true, y_pred, title='混淆矩阵'):
    """绘制混淆矩阵。"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[-1, 0, 1], yticklabels=[-1, 0, 1])
    plt.title(title)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()

# --- 主函数 ---

def main():
    # --- 1. 环境与设备检查 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}")

    # --- 2. 数据加载与准备 ---
    print("--- 1. 开始加载数据 ---")
    main_df = load_data(TRAIN_DATA_DIR, sample_size=SAMPLE_TRAIN_FILES, description="加载训练/验证数据")
    
    label_col = 'label'
    features_cols = [col for col in main_df.columns if col not in ['tradingtime', 'symbol', 'label']]
    main_df = main_df[features_cols + [label_col]].dropna()

    # --- 3. 数据集划分与标准化 ---
    print("\n--- 2. 划分数据集并进行标准化 ---")
    val_size = int(len(main_df) * VALIDATION_SET_RATIO)
    train_df = main_df.iloc[:-val_size]
    val_df = main_df.iloc[-val_size:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_features = scaler.fit_transform(train_df[features_cols])
    val_features = scaler.transform(val_df[features_cols])
    
    train_labels = train_df[label_col].values + 1
    val_labels = val_df[label_col].values + 1
    
    # --- 4. 创建PyTorch数据集和加载器 ---
    print("\n--- 3. 创建PyTorch数据集和加载器 ---")
    train_dataset = TimeSeriesDataset(train_features, train_labels, SEQUENCE_LENGTH)
    val_dataset = TimeSeriesDataset(val_features, val_labels, SEQUENCE_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- 5. 初始化模型、损失函数和优化器 ---
    print("\n--- 4. 初始化模型 ---")
    input_dim = len(features_cols)
    hidden_dim = 64
    layer_dim = 2
    output_dim = 3
    dropout_prob = 0.2

    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, dropout_prob).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(model)

    # --- 6. 训练循环 ---
    print("\n--- 5. 开始模型训练 ---")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 5
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss, total_train_correct, total_train_samples = 0, 0, 0
        
  
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train_samples += labels.size(0)
            total_train_correct += (predicted == labels).sum().item()
           

        avg_train_loss = total_train_loss / total_train_samples
        avg_train_acc = total_train_correct / total_train_samples
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)

        # 验证循环
        model.eval()
        total_val_loss, total_val_correct, total_val_samples = 0, 0, 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item() * sequences.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val_samples += labels.size(0)
                total_val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = total_val_loss / total_val_samples
        avg_val_acc = total_val_correct / total_val_samples
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

        # 早停与模型保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_lstm_model_pytorch.pth')
            epochs_no_improve = 0
            print("  -> 验证集Loss改善，保存模型。")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"  -> 验证集Loss连续{patience}轮未改善，提前停止训练。")
                break
    
    # --- 7. 结果评估 ---
    print("\n--- 6. 评估最佳模型性能 ---")
    plot_history(history)
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_lstm_model_pytorch.pth'))
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    print("\n--- 在'未来时段'验证集上的表现 ---")
    print("分类报告:")
    print(classification_report(np.array(all_labels) - 1, np.array(all_preds) - 1))
    plot_confusion_matrix_torch(np.array(all_labels) - 1, np.array(all_preds) - 1, title='未来时段验证集 - 混淆矩阵')

if __name__ == "__main__":
    main()
