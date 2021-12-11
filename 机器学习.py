import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.neighbors import KNeighborsClassifier # k nearest neighbours
from sklearn.model_selection import  GridSearchCV #, StratifiedKFold, train_test_split
#     from sklearn.preprocessing import MinMaxScaler


# 读取训练数据
data = pd.read_csv('C:\\Users\\a_fool_zzzzz\\Desktop\\机器学习课设\\数据集\\train.csv')
data

# 读取测试数据
validate_data = pd.read_csv('C:\\Users\\a_fool_zzzzz\\Desktop\\机器学习课设\\数据集\\test.csv')
validate_data

# 将测试数据的索引单独导出
validate_data_ids = validate_data['id']


# 对训练数据预处理
# 查看训练数据的数据种类
data.dtypes

# 检查训练数据是否有缺省
data.isnull().any()

# 绘制所有特征分布的样本数量情况
for i in ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul']:
    plt.hist(data[i])
    plt.title(i)
    plt.show()

# 绘制数据热力图查看特征之间的相关性(将id列删除)
sns.heatmap(data.drop(['id'], 1, errors='ignore').corr())

# 绘制不同类型的样本所占样本数量
sns.countplot(x='type', data=data)

# 绘制每个类型中颜色特征所占的样本数量
sns.countplot(x='type', hue='color', data=data, palette=['yellow', 'green', 'black', 'grey', 'blue', 'red'])

sns.pairplot(data, hue='type')

# 将颜色特征数据归一化
# colormap = {name: idx for idx, name in enumerate(data['color'].astype('category').cat.categories )}
# ncolors = len(colormap) - 1
# data['color'] = data['color'].apply(
#     lambda x: colormap[x] / ncolors
# )
# validate_data['color'] = validate_data['color'].apply(
#     lambda x: colormap[x] / ncolors
# )

# 由于颜色特征区别不大所以将其直接删除
def preprocess_data(data, validate_data):
    data.drop(['id', 'color', ''], errors='ignore', axis=1, inplace=True)
    validate_data.drop(['id', 'color'], errors='ignore', axis=1, inplace=True)

preprocess_data(data, validate_data)

# 检查剩余数据
data.dtypes
data, validate_data

# 将训练集拆分成特征数据和标记结果数据
train_set_x, train_set_y = data.drop('type', 1), data['type']

# 训练模型为网格搜索使用k近邻投票算法进行训练
classifier = GridSearchCV(
    KNeighborsClassifier(),
    param_grid={
        'n_neighbors': np.arange(1, 100),
        'p': np.arange(1, 10)
    }, # 训练参数为选择的最近样本数n_neighbors和闵可夫斯基距离的参数p
    scoring='accuracy', # 正确结果为样本预测结果和标记结果相同
    cv=3 # 使用3折交叉验证
)

# 将训练数据传入并计算其每一中参数组合之后得到得正确率以及正确率的均值和最大值
classifier.fit(train_set_x, train_set_y)

scores = classifier.cv_results_['mean_test_score']
scores, scores.mean(), scores.max()

# 查看最佳参数
classifier.best_params_

# 计算将使用最佳参数时的训练数据的正确率平均值
np.mean(classifier.predict(train_set_x) == train_set_y)

# 使用训练出来的模型对测试数据进行判断并写入文件
submission = classifier.predict(validate_data)

pd.DataFrame({'id': validate_data_ids, 'type': submission}).to_csv('C:\\Users\\a_fool_zzzzz\\Desktop\\机器学习课设\\数据集\\submission.csv', index=False)