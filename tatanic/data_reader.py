import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def read():
    data = pd.read_csv("JQXX/tatanic/train.csv")
    label = data["Survived"].values.reshape(-1, 1)
    data = data.drop(["PassengerId", "Cabin", "Name", "Ticket"], axis=1)
    # 处理数据中的空值
    age = int(np.mean(data["Age"].fillna(0).values))
    value = {"Age": age, "Embarked": "un"}
    data = data.fillna(value)
    cols = ["Age", "SibSp", "Parch", "Fare"]
    # 将数据中的字符串，转换为数值
    coder = LabelEncoder()
    ont = OneHotEncoder(categories='auto')
    temp = data[["Sex", "Embarked"]].apply(lambda item: coder.fit_transform(item))
    # 使用二值化对离散值进行标准化
    temp = ont.fit_transform(temp.values).astype(int).toarray()
    data = data[cols].apply(lambda item: (item - np.min(item)) / (np.max(item) - np.min(item)), axis=0).values
    data = np.hstack((data, temp))
    # 将1添加到data的第1列
    ones = np.ones((data.shape[0], data.shape[1] + 1))
    ones[:, 1:] = data
    return ones, label

    # 展示数据之间的关系
    # cm = np.corrcoef(data.values.T)
    # sns.set(font_scale=1.5)
    # sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
    # plt.show()
