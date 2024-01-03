import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 展平嵌套字典的函数
def flatten_nested_dict(d):
    flat_dict = {}
    for outer_key, inner_dict in d.items():
        for inner_key, value in inner_dict.items():
            flat_dict[f'{outer_key}_{inner_key}'] = value
    return flat_dict

# 加载数据
# 假设您的DataFrame名为df，其中包含特征和一个名为'label'的列作为目标标签
# df = ...



# 分割特征和标签
X = df.drop('label', axis=1)  # 特征
y = df['label']  # 标签

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
clf.fit(X_train, y_train)

# 预测测试集
y_pred = clf.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
