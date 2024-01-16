import pandas as pd
from datetime import datetime

# 读取txt文件并转换为DataFrame
def read_txt_convert_to_df(file_path):
    df = pd.read_csv(file_path, sep='[,;]', usecols=[0, 1, 2, 3, 4, 5])
    df.columns = ['timestamp', 'ms', 'ppg', 'acc_x', 'acc_y', 'acc_z']
    df['timestamp'] = df['timestamp']*1000 + df['ms']
    df['formatted_time'] = pd.to_datetime(df['timestamp'], unit='s')
    df['formatted_time'] = df['formatted_time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
    return df

# 将DataFrame写入CSV文件
def df_to_csv(df, output_path):
    df.to_csv(output_path, index=False)

# 示例使用
file_path = 'log0.txt'
output_path = 'formatDate.csv'

df = read_txt_convert_to_df(file_path)
df_to_csv(df, output_path)
