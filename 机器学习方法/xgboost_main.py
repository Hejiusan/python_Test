import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn
from hrvanalysis import remove_ectopic_beats
from hrvanalysis import interpolate_nan_values
from hrvanalysis import remove_outliers, extract_features
from scipy.interpolate import PchipInterpolator
from hrvanalysis import get_time_domain_features
from hrvanalysis import get_frequency_domain_features
from hrvanalysis import get_poincare_plot_features
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt

from QRSDetectorOffline import QRSDetectorOffline
class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)  # 如果是int64类型，转换为Python的标准整数类型int
        return json.JSONEncoder.default(self, obj)


# 展平嵌套字典的函数
def flatten_nested_dict(d):
    flat_dict = {}
    for outer_key, inner_dict in d.items():
        for inner_key, value in inner_dict.items():
            flat_dict[f'{outer_key}_{inner_key}'] = value
    return flat_dict


def plotHeatMap(Y_test, Y_pred):
    con_mat = confusion_matrix(Y_test, Y_pred)
    # 绘图
    plt.figure(figsize=(4, 5))
    seaborn.heatmap(con_mat, annot=True, fmt='.20g', cmap='Blues')
    plt.ylim(0, 5)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

def getDataSet(record):

    rr_intervals_without_outliers = remove_outliers(rr_intervals=record['rrInterval'],low_rri=300, high_rri=2000)
    interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers,interpolation_method="linear")
    # 删除相邻的两个 RRI 值之间的差异大于 20% 的异常搏动

    record['process'] = remove_ectopic_beats(rr_intervals=interpolated_rr_intervals,
                                                   method='malik', custom_removing_rule=0.2, verbose=True)

    # 提取 RRI 列数据
    rri_data = record['process'].dropna()

    # 创建 PCHIP 插值对象
    interpolator = PchipInterpolator(rri_data.index, rri_data)

    # 在原始索引上进行插值
    record['rrInterval'] = interpolator(record.index)

    rri = record['rrInterval']
    time_domain_features = get_time_domain_features(rri)
    frequency_domain_features = get_frequency_domain_features(rri)
    poincare_plot_features = get_poincare_plot_features(rri)
    # 合并为一个大字典
    merged_data = {
        "time_domain": time_domain_features,
        "frequency_domain": frequency_domain_features,
        "poincare_plot": poincare_plot_features
    }

    return merged_data



    # 将int64类型转换为Python的标准整数类型int
    # merged_data = merged_data.tolist()
    # 将字典转换为JSON字符串 自定义编码格式 转化int64-》int
    json_data = json.dumps(merged_data, cls=NumpyEncoder, indent=4)  # indent参数用于美化输出，非必需
    # file_path = "features/" + n + '.json'
    # # 将JSON字符串写入文件
    # with open(file_path, "w") as json_file:
    #     json_file.write(json_data)
    # print(f"JSON数据已保存到文件：{file_path}")

folder_path = 'hrv-5min'
file_names = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]


# 初始化一个空的DataFrame
df = pd.DataFrame()
for n in file_names:
    # 读取心电数据记录
    print("正在读取 " + n)
    # 读取数据
    record = pd.read_csv('hrv-5min/' + n)
    flat_dict = flatten_nested_dict(getDataSet(record))
    flat_dict['label'] = 1
    df = df._append(flat_dict, ignore_index=True)


# df_102 = pd.read_csv('860911061000102-2023-09-02.csv')
df_102 = pd.read_csv('860911061000102-2023-09-02.csv')
# 将时间戳转换为日期时间
df_102['datetime'] = pd.to_datetime(df_102['timestamp'], unit='ms')

# 设置五分钟间隔分组
df_grouped = df_102.groupby(pd.Grouper(key='datetime', freq='5Min'))

# 对每个分组进行处理
total_groups = len(list(df_grouped))
current_group = 0
for name, group in df_grouped:
    current_group += 1
    if current_group == 1 or current_group == total_groups:
        continue  # 跳过第一个和最后一个分组
    # 这里可以对每个五分钟内的数据进行处理
    if (group.size == 0):
        continue
    flat_dict = flatten_nested_dict(getDataSet(group))
    flat_dict['label'] = 0
    df = df._append(flat_dict, ignore_index=True)



# print(df)

# 分割特征和标签
X = df.drop('label', axis=1)  # 特征
y = df['label']  # 标签

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 创建一个XGBoost分类器的实例
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# 利用训练集样本对分类器模型进行训练
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# 画出训练后模型的混淆矩阵，方便观察训练的效果
plotHeatMap(y_test,y_pred)


# # 创建随机森林模型
# clf = RandomForestClassifier(random_state=42, n_estimators=120)
#
# # 训练模型
# clf.fit(X_train, y_train)
#
# # 预测测试集
# y_pred = clf.predict(X_test)

# 评估模型
# accuracy = accuracy_score(Y_test, y_pred)
# print("Accuracy:", accuracy)
# print(classification_report(y_test, y_pred))