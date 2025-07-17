import os
import pandas as pd
import numpy as np
from scipy.stats import zscore

def cal_TF(df):
    """
    计算高频交易因子
    """
    
    # 1. 价格动量因子 (0.5s级别的收益率)
    df['momentum_1s'] = df['LASTPRICE'].pct_change(periods=2)  # 2*0.5s=1s
    
    # 2. 订单簿不平衡因子 (买卖压力)
    df['bid_volume'] = df[['BUYVOLUME01', 'BUYVOLUME02', 'BUYVOLUME03']].sum(axis=1)
    df['ask_volume'] = df[['SELLVOLUME01', 'SELLVOLUME02', 'SELLVOLUME03']].sum(axis=1)
    df['order_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'] + 1e-6)
    
    # 3. 流动性因子 (价差)
    df['spread'] = (df['SELLPRICE01'] - df['BUYPRICE01']) / df['LASTPRICE']
    
    # 4. 成交量突变因子
    df['volume_change'] = df['TRADEVOLUME'].pct_change()
    
    # 5. 波动率因子 (短期波动)
    df['ret'] = df['LASTPRICE'].pct_change()
    df['volatility'] = df['ret'].rolling(window=10, min_periods=3).std()
    
    # 6. 资金流因子
    df['money_flow'] = df['LASTPRICE'] * df['TRADEVOLUME']
    df['money_flow_sma'] = df['money_flow'].rolling(window=5).mean()
    
    # 7. 价格加速度因子
    df['price_acc'] = df['momentum_1s'].diff()
    
    # 8. 买卖价量相关性
    df['bid_ask_corr'] = df['bid_volume'].rolling(window=5).corr(df['ask_volume'])
    
    # 标准化因子
    factor_cols = ['momentum_1s', 'order_imbalance', 'spread', 'volume_change', 
                  'volatility', 'money_flow_sma', 'price_acc', 'bid_ask_corr']
    
    for col in factor_cols:
        if col in df.columns:
            df[col+'_z'] = zscore(df[col].replace([np.inf, -np.inf], np.nan).fillna(0))
    
    # TODO: how to combine those factor, maybe by linear regression?
    df['composite_factor'] = df[[c+'_z' for c in factor_cols]].mean(axis=1)
    
    return df

def generate_signals(df):
    """
    基于复合因子生成交易信号
    """
    # 简单阈值策略
    df['signal'] = 0
    df.loc[df['composite_factor'] > 1.0, 'signal'] = 1   # 买入信号
    df.loc[df['composite_factor'] < -1.0, 'signal'] = -1  # 卖出信号
    
    # 确保信号是离散的 (0.5s级别可能需要平滑处理)
    df['signal'] = df['signal'].rolling(window=3, min_periods=1).median()
    
    return df

if __name__ == '__main__':
    train_dir = 'train'
    train_dates = sorted(os.listdir(train_dir))[1:]

    index = 0
    
    data_dict = {}
    data_dict['IC'] = []
    data_dict['IF'] = []
    data_dict['IH'] = []
    data_dict['IM'] = []

    for date in train_dates:
        index += 1
        if index > 20:
            break
        path = f"{train_dir}/{date}"
        name = os.listdir(path)
        Mfiles = [n for n in name if 'M.parquet' in n]
        for f in Mfiles:
            df = pd.read_parquet(f'{path}/{f}')
            
            # 计算因子
            df = cal_TF(df)
            
            # 生成信号
            df = generate_signals(df)
            
            data_dict[f[:2]].append(df) # f[:2] = 'IC' or 'IM' e.t.c.
        
        print(f"Processed date: {date}")