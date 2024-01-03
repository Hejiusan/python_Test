import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
from hrvanalysis import remove_ectopic_beats
from hrvanalysis import interpolate_nan_values
from hrvanalysis import remove_outliers, extract_features
from scipy.interpolate import PchipInterpolator
from hrvanalysis import get_time_domain_features
from hrvanalysis import get_frequency_domain_features
from hrvanalysis import get_poincare_plot_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

def getDataSet(number):
    # 读取心电数据记录
    print("正在读取 " + number + " 号心电数据...")
    # 读取数据
    record = pd.read_csv('hrv-5min/' + number + '.csv')
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


numberSet = ['001_1_s', '001_2_s', '002_1_s', '002_2_s', '003_1_s', '003_2_s', '004_1_s', '005_1_s', '005_2_s', '006_1_s',
             '007_1_s', '007_2_s', '008_1_s', '010_1_s', '011_1_s', '012_1_s', '013_1_s', '014_1_s', '015_1_s', '016_1_s',
             '017_1_s', '019_1_s10', '020_1_s5', '022_1_s5', '023_1_s10', '024_1_s10', '026_1_s10', '027_1_s10', '028_1_s5',
             '029_1_s10', '030_1_s10']

# 初始化一个空的DataFrame
df = pd.DataFrame()
for n in numberSet:
    flat_dict = flatten_nested_dict(getDataSet(n))
    df = df._append(flat_dict, ignore_index=True)

df['label'] = [3,3,3,3,3,3,1,1,1,3,
               3,3,1,1,1,3,2,1,1,1,
               1,1,1,3,1,1,1,2,3,1,3]

# print(df)

# 分割特征和标签
X = df.drop('label', axis=1)  # 特征
y = df['label']  # 标签

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林模型
clf = RandomForestClassifier(random_state=42, n_estimators=120)

# 训练模型
clf.fit(X_train, y_train)

# 预测测试集
y_pred = clf.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))