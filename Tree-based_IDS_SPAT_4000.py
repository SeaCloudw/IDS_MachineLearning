#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import plot_importance
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold

from sklearn.feature_selection import SelectKBest,f_classif
# !!!!!!!注意以下都是有监督学习算法，传入的数据集已经打好label,在最后一列
# 所以直接fit()->predict()
# plot热力图大部分注释掉了

# ## Machine learning model training

# ### Training four base learners: decision tree, random forest, extra trees, XGBoost
# 决策树、随机森林、额外树和 XGBoost
# 传统机器学习模型：这是现成的学习模型，调用 fit 方法后，模型会自动进行参数调整，速度通常较快
# 深度学习模型：需要定义训练轮次（epochs），每个 epoch 表示模型在整个训练数据集上完整训练一次，训练速度较慢，
def preprocess(filepath,train_size):
    """
    该函数用于预处理数据集并划分训练集和测试集。数据填充部分被省略

    参数:
    :param filepath: str, 数据集的路径
    :param train_size: float, 训练集占整个数据集的比例，一般为0.8

    返回：
    :X_train:训练集特征矩阵
    :X_test: 测试集特征矩阵
    :y_train:训练集标签向量
    :y_test:测试集标签向量

    """
    df = pd.read_csv(filepath)

    # Min-max normalization
    # 标准化处理，并填充缺失值:numeric_features 是一个包含所有数值特征列名的索引对象

    numeric_features = df.dtypes[df.dtypes != 'object'].index
    # 缩放到 [0, 1] 之间
    # df[numeric_features] = df[numeric_features].apply(
    #     lambda x: (x - x.min()) / (x.max()-x.min()))
    # Fill empty values by 0将数据集中所有缺失值（NaN）填充为 0。
    df = df.fillna(0)


    # ### split train set and test set
    # 处理标签（这里最后一列是标签）
    """ 
    print("需要处理的标签\n",df.iloc[:5:,-1])
    print("需要处理的标签\n",df.iloc[-5:,-1])
    print("需要处理的标签\n",df.iloc[:5:,-1])
    0    normal
    1    normal
    2    normal
    3    normal
    4    norma
    344    DoS
    345    DoS
    346    DoS
    347    DoS
    348    DoS
    Name: Label, dtype: object """

    labelencoder = LabelEncoder()
    df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])


    """ LabelEncoder 会为这些标签值分配整数编号，编号的顺序取决于标签值在数据集中首次出现的顺序。例如：

    BENIGN -> 0
    WebAttack -> 1
    Bot -> 2
    Infiltration -> 3
    DoS -> 4
    PortScan -> 5
    BruteForce -> 6 """

    """ 
    print("编码后的标签\n", df.iloc[0:5, -1])
    print("编码后的标签\n", df.iloc[-5:, -1])
    编码后的标签
    0    1
    1    1
    2    1
    3    1
    4    1

    344    0
    345    0
    346    0
    347    0
    348    0
    Name: Label, dtype: int32 """
    # x是特征 矩阵,先是删除label这一列，.values将其转成NumPy 数组
    # X = df.drop(['Label','Type','ID','AID'],axis=1).values 
    X = df.drop(['Label','Type','ID','Modulation','AID'],axis=1).values 

    # y是标签向量，就是最后一列，reshape数组维数转换，这里是一维数组转换为二维数组(-1表示自动计算 行数，  1表示一共1列)
    y = df.iloc[:, -1].values.reshape(-1,1)
    # print(y[:5])
    # 展为一维数组,一开始的df.iloc[:, -1].values就是一维的，所以有点多余
    y=np.ravel(y)
   
    # 划分测试集与数据集NumPy 数组类型；train_size=0.8 表示训练集占总数据的 80%。
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size =train_size , test_size = 1-train_size, random_state = 0,stratify = y)

    """     
    # 查看训练集特征矩阵的前几行
    print("训练集特征矩阵 (X_train) 前几行:")
    print(pd.DataFrame(X_train).head())

    # 查看训练集标签向量的前几行
    print("\n训练集标签向量 (y_train) 前几行:")
    print(y_train[:5])

    # 查看训练集的形状

    print("\n训练集特征矩阵 (X_train) 形状:", X_train.shape)
    print("\n测试集特征矩阵 形状:", X_test.shape)
    print("训练集标签向量 (y_train) 形状:", y_train.shape)
    #查看标签向量的分布情况
    print(pd.Series(y_train).value_counts()) 
    """
    """ 
    索引是标签值，值是每个标签值对应的样本数量。
    1    239
    0     40
    dtype: int64
    """
    # ### Oversampling by SMOTE
    # 过采样，解决数据集中的类别不平衡问题
    # n_jobs=-1：使用所有可用的 CPU 核心来加速计算。

    # sampling_strategy={4: 1500}：指定对少数类（标签为 4）进行过采样，生成 1500 个样本,注意这里是最终由1500个。
    # 因为其那面看到4在样本里是最少的

    # 增加0 加到200
    # smote=SMOTE(n_jobs=-1,sampling_strategy={0:1600,1:1600})  

    # # 对训练集进行过采样，生成新的训练集
    # X_train, y_train = smote.fit_resample(X_train, y_train)

    # 查看过采样后的标签分布情况
    print(pd.Series(y_train).value_counts())
    print(pd.Series(y_test).value_counts())
    return     X_train, X_test, y_train, y_test


def preprocess_feature_filter(filepath,train_size):
    """
    该函数用于预处理数据集并划分训练集和测试集。数据填充部分被省略

    参数:
    :param filepath: str, 数据集的路径
    :param train_size: float, 训练集占整个数据集的比例，一般为0.8

    返回：
    :X_train:训练集特征矩阵
    :X_test: 测试集特征矩阵
    :y_train:训练集标签向量
    :y_test:测试集标签向量

    """
    df = pd.read_csv(filepath)

    # Min-max normalization
    # 标准化处理，并填充缺失值:numeric_features 是一个包含所有数值特征列名的索引对象

    numeric_features = df.dtypes[df.dtypes != 'object'].index
    # 缩放到 [0, 1] 之间
    df[numeric_features] = df[numeric_features].apply(
        lambda x: (x - x.min()) / (x.max()-x.min()))
    # Fill empty values by 0将数据集中所有缺失值（NaN）填充为 0。
    df = df.fillna(0)


    # ### split train set and test set
    # 处理标签（这里最后一列是标签）
    """ 
    print("需要处理的标签\n",df.iloc[:5:,-1])
    print("需要处理的标签\n",df.iloc[-5:,-1])
    print("需要处理的标签\n",df.iloc[:5:,-1])
    0    normal
    1    normal
    2    normal
    3    normal
    4    norma
    344    DoS
    345    DoS
    346    DoS
    347    DoS
    348    DoS
    Name: Label, dtype: object """

    labelencoder = LabelEncoder()
    df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])


    """ LabelEncoder 会为这些标签值分配整数编号，编号的顺序取决于标签值在数据集中首次出现的顺序。例如：

    BENIGN -> 0
    WebAttack -> 1
    Bot -> 2
    Infiltration -> 3
    DoS -> 4
    PortScan -> 5
    BruteForce -> 6 """

    """ 
    print("编码后的标签\n", df.iloc[0:5, -1])
    print("编码后的标签\n", df.iloc[-5:, -1])
    编码后的标签
    0    1
    1    1
    2    1
    3    1
    4    1

    344    0
    345    0
    346    0
    347    0
    348    0
    Name: Label, dtype: int32 """
    # x是特征 矩阵,先是删除label这一列，.values将其转成NumPy 数组
    # X = df.drop(['Label','Type','ID','AID'],axis=1).values 
    X = df.drop(['Label','Type','ID','Modulation','AID'],axis=1).values 
    # y是标签向量，就是最后一列，reshape数组维数转换，这里是一维数组转换为二维数组(-1表示自动计算 行数，  1表示一共1列)
    y = df.iloc[:, -1].values.reshape(-1,1)
    # print(y[:5])
    # 展为一维数组,一开始的df.iloc[:, -1].values就是一维的，所以有点多余
    y=np.ravel(y)

    # 过滤，方差    
    sel=VarianceThreshold(0.05)
    sel.fit(X)   #获得方差，不需要y
    print('Variances is %s'%sel.variances_)
    print('After transform is \n%s'%sel.transform(X))
    print('The surport is %s'%sel.get_support(True))#如果为True那么返回的是被选中的特征的下标
    print('The surport is %s'%sel.get_support(False))#如果为FALSE那么返回的是布尔类型的列表，反应是否选中这列特征
    print('len The surport is %d'%len(sel.get_support(True)))#如果为True那么返回的是被选中的特征的下标
    # print(X.shape())
    X_new=sel.fit_transform(X)
    X=X_new
    # （17） 

    """     # 特征过滤,单变量特征
    #  使用F值进行特征选择，选前十个
    sel=SelectKBest(score_func=f_classif,k=10)
    sel.fit(X,y)  #计算统计指标，这里一定用到y
    # print('scores_:\n',sel.scores_)
    # print('pvalues_:',sel.pvalues_)
    print('selected index:',sel.get_support(True))
    # selected index: [ 0  1  4  9 11 13 16 17 18 21]
    # print('after transform:\n',sel.transform(X))
    X=sel.transform(X)

    """
   
    # 划分测试集与数据集NumPy 数组类型；train_size=0.8 表示训练集占总数据的 80%。
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size =train_size , test_size = 1-train_size, random_state = 0,stratify = y)
    print("\n测试集特征矩阵 形状:", X_test.shape)
    """     
    # 查看训练集特征矩阵的前几行
    print("训练集特征矩阵 (X_train) 前几行:")
    print(pd.DataFrame(X_train).head())

    # 查看训练集标签向量的前几行
    print("\n训练集标签向量 (y_train) 前几行:")
    print(y_train[:5])

    # 查看训练集的形状

    print("\n训练集特征矩阵 (X_train) 形状:", X_train.shape)
    print("\n测试集特征矩阵 形状:", X_test.shape)
    print("训练集标签向量 (y_train) 形状:", y_train.shape)
    #查看标签向量的分布情况
    print(pd.Series(y_train).value_counts()) 
    """
    """ 
    索引是标签值，值是每个标签值对应的样本数量。
    1    239
    0     40
    dtype: int64
    """
    # ### Oversampling by SMOTE
    # 过采样，解决数据集中的类别不平衡问题
    # n_jobs=-1：使用所有可用的 CPU 核心来加速计算。

    # sampling_strategy={4: 1500}：指定对少数类（标签为 4）进行过采样，生成 1500 个样本,注意这里是最终由1500个。
    # 因为其那面看到4在样本里是最少的

    # 增加0 加到200
    smote=SMOTE(n_jobs=-1,sampling_strategy={0:1600,1:1600})  

    # 对训练集进行过采样，生成新的训练集
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # 查看过采样后的标签分布情况
    print(pd.Series(y_train).value_counts())
    print(pd.Series(y_test).value_counts())
    return     X_train, X_test, y_train, y_test

def test1_Decisiontree(X_train, X_test, y_train, y_test):
    """
    该函数用于执行决策树模型。

    参数:
    :X_train:训练集特征矩阵
    :X_test: 测试集特征矩阵
    :y_train:训练集标签向量
    :y_test:测试集标签向量

    返回：
    :dt:Decisiontree模型
    :dt_train=dt.predict(X_train)
    :dt_test=dt.predict(X_test)

    """
  


    # Decision tree training and prediction
    # 决策树分类器
    dt = DecisionTreeClassifier(random_state = 0)
    #传入数据
    dt.fit(X_train,y_train) 
    # 计算模型在测试集上的准确率
    dt_score=dt.score(X_test,y_test)
    # 这是预测标签,对测试集X_test预测
    y_predict=dt.predict(X_test)
    # 这是真实标签
    y_true=y_test
    print('Accuracy of DT: '+ str(dt_score))
    # 计算并打印模型的精确度、召回率和 F1 分数
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    print('Precision of DT: '+(str(precision)))
    print('Recall of DT: '+(str(recall)))
    print('F1-score of DT: '+(str(fscore)))
    print(classification_report(y_true,y_predict))
    # 绘制混淆矩阵，可视化模型的预测结果，使用 Seaborn 库绘制热力图
    cm=confusion_matrix(y_true,y_predict)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    # plt.show()


    dt_train=dt.predict(X_train)
    dt_test=dt.predict(X_test)
    return dt,dt_train,dt_test
def   test2_Random_Forest(X_train, X_test, y_train, y_test):

    # Random Forest training and prediction
    rf = RandomForestClassifier(random_state = 0)
    rf.fit(X_train,y_train) 
    rf_score=rf.score(X_test,y_test)
    y_predict=rf.predict(X_test)
    y_true=y_test
    print('Accuracy of RF: '+ str(rf_score))
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    print('Precision of RF: '+(str(precision)))
    print('Recall of RF: '+(str(recall)))
    print('F1-score of RF: '+(str(fscore)))
    print(classification_report(y_true,y_predict))
    cm=confusion_matrix(y_true,y_predict)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    # plt.show()

 

    rf_train=rf.predict(X_train)
    rf_test=rf.predict(X_test)
    return rf,rf_train,rf_test



def   test3_Extra_trees(X_train, X_test, y_train, y_test):

    # Extra trees training and prediction
    et = ExtraTreesClassifier(random_state = 0)
    et.fit(X_train,y_train) 
    et_score=et.score(X_test,y_test)
    y_predict=et.predict(X_test)
    y_true=y_test
    print('Accuracy of ET: '+ str(et_score))
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    print('Precision of ET: '+(str(precision)))
    print('Recall of ET: '+(str(recall)))
    print('F1-score of ET: '+(str(fscore)))
    print(classification_report(y_true,y_predict))
    cm=confusion_matrix(y_true,y_predict)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    # plt.show()


    # In[16]:


    et_train=et.predict(X_train)
    et_test=et.predict(X_test)

    return et,et_train,et_test

def   test4_XGboost (X_train, X_test, y_train, y_test):
    # XGboost training and prediction
    xg = xgb.XGBClassifier(n_estimators = 10)
    xg.fit(X_train,y_train)
    xg_score=xg.score(X_test,y_test)
    y_predict=xg.predict(X_test)
    y_true=y_test
    print('Accuracy of XGBoost: '+ str(xg_score))
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    print('Precision of XGBoost: '+(str(precision)))
    print('Recall of XGBoost: '+(str(recall)))
    print('F1-score of XGBoost: '+(str(fscore)))
    print(classification_report(y_true,y_predict))
    cm=confusion_matrix(y_true,y_predict)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    # plt.show()


    # In[18]:


    xg_train=xg.predict(X_train)
    xg_test=xg.predict(X_test)
    return xg,xg_train,xg_test


def Stacking_4_model(dt_train, et_train, rf_train, xg_train,dt_test, et_test, rf_test, xg_test,y_train,y_test,X_test) :
    # ### Stacking model construction (ensemble for 4 base learners)
    # 将每个基础模型在训练集上的预测结果组合成一个新的数据集,堆叠（Stacking）

    # In[19]:


    # Use the outputs of 4 base models to construct a new ensemble model
    base_predictions_train = pd.DataFrame( {
        'DecisionTree': dt_train.ravel(),
            'RandomForest': rf_train.ravel(),
        'ExtraTrees': et_train.ravel(),
        'XgBoost': xg_train.ravel(),
        })
    print(base_predictions_train.head(5))


    # In[20]:

    dt_train=dt_train.reshape(-1, 1)
    et_train=et_train.reshape(-1, 1)
    rf_train=rf_train.reshape(-1, 1)
    xg_train=xg_train.reshape(-1, 1)
    dt_test=dt_test.reshape(-1, 1)
    et_test=et_test.reshape(-1, 1)
    rf_test=rf_test.reshape(-1, 1)
    xg_test=xg_test.reshape(-1, 1)


    # In[21]:

    # 列向量，按列组合，每一个train当做新的一列，得到新的数据集
    x_train = np.concatenate(( dt_train, et_train, rf_train, xg_train), axis=1)
    x_test = np.concatenate(( dt_test, et_test, rf_test, xg_test), axis=1)
    # x_test=X_test


    # In[22]:

    # 新的训练集
    stk = xgb.XGBClassifier().fit(x_train, y_train)


    # In[23]:

    y_predict=stk.predict(x_test)
    y_true=y_test
    stk_score=accuracy_score(y_true,y_predict)
    print('Accuracy of Stacking: '+ str(stk_score))
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    print('Precision of Stacking: '+(str(precision)))
    print('Recall of Stacking: '+(str(recall)))
    print('F1-score of Stacking: '+(str(fscore)))
    print(classification_report(y_true,y_predict))
    cm=confusion_matrix(y_true,y_predict)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    # plt.show()

def reprocess_feature_selection(filepath,train_size,dt,rf,et,xg): 
    df = pd.read_csv(filepath)

    # Min-max normalization
    # 标准化处理，并填充缺失值:numeric_features 是一个包含所有数值特征列名的索引对象

    numeric_features = df.dtypes[df.dtypes != 'object'].index
    # 缩放到 [0, 1] 之间
    df[numeric_features] = df[numeric_features].apply(
        lambda x: (x - x.min()) / (x.max()-x.min()))
    # Fill empty values by 0将数据集中所有缺失值（NaN）填充为 0。
    df = df.fillna(0)


    # ### split train set and test set
    # 处理标签（这里最后一列是标签）
    """ 
    print("需要处理的标签\n",df.iloc[:5:,-1])
    print("需要处理的标签\n",df.iloc[-5:,-1])
    print("需要处理的标签\n",df.iloc[:5:,-1])
    0    normal
    1    normal
    2    normal
    3    normal
    4    norma
    344    DoS
    345    DoS
    346    DoS
    347    DoS
    348    DoS
    Name: Label, dtype: object """

    labelencoder = LabelEncoder()
    df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
    # x是特征 矩阵,先是删除label这一列，.values将其转成NumPy 数组
    X = df.drop(['Label','Type','ID'],axis=1).values 
    # y是标签向量，就是最后一列，reshape数组维数转换，这里是一维数组转换为二维数组(-1表示自动计算 行数，  1表示一共1列)
    y = df.iloc[:, -1].values.reshape(-1,1)
    # print(y[:5])
    # 展为一维数组,一开始的df.iloc[:, -1].values就是一维的，所以有点多余
    y=np.ravel(y)
   

    # ## Feature Selection

    # ### Feature importance
    # Save the feature importance lists generated by four tree-based algorithms
    # !!!调整重要性权重
    dt_feature = dt.feature_importances_
    rf_feature = rf.feature_importances_
    et_feature = et.feature_importances_
    xgb_feature = xg.feature_importances_

    # calculate the average importance value of each feature
    avg_feature = (dt_feature + rf_feature + et_feature + xgb_feature)/4
    # .colums取列名，并且把列名转换成数组
    feature=(df.drop(['Label'],axis=1)).columns.values
    print ("Features sorted by their score:")
    #   zip返回一个包含元组的列表。每个元组的形式为 (重要性, 特征名称)  sort按重要性排序
    # print (sorted(zip(map(lambda x: round(x, 4), avg_feature), feature), reverse=True))
    f_list = sorted(zip(map(lambda x: round(x, 4), avg_feature), feature), reverse=True)

    # Select the important features from top-importance to 
    # bottom-importance until the accumulated importance reaches 0.9 (out of 1)
    Sum = 0
    fs = []#fs存储被选中的特征名，最重要的一些
    for i in range(0, len(f_list)):
        Sum = Sum + f_list[i][0]
        fs.append(f_list[i][1])
        if Sum>=0.9:
            break        


    print("\n原本的规模\n",df.shape)
    X_fs = df[fs].values
    print("\n按照重要性修改选择后的规模\n",X_fs.shape)
    """ 原本的规模
     (349, 30)

    按照重要性修改选择后的规模
     (349, 9) """
    # In[31]:

    #划分训练集与验证集
    X_train, X_test, y_train, y_test = train_test_split(X_fs,y, train_size = train_size, test_size = 1-train_size, random_state = 0,stratify = y)


    # print(pd.Series(y_train).value_counts())

    # # ### Oversampling by SMOTE

    # # In[34]:


    # # from imblearn.over_sampling import SMOTE
    # smote=SMOTE(n_jobs=-1,sampling_strategy={4:1500})


    # # In[35]:


    # X_train, y_train = smote.fit_resample(X_train, y_train)


    # # In[36]:

    # print("\n过采样后的分布")
    # print(pd.Series(y_train).value_counts())

    # print("以下是根据新的数据集生成的结果（重要性权重）")
    return X_train, X_test, y_train, y_test

def main():
    # filepath='./data/SPAT_DoS.csv'
    # filepath='./data/datal_Dos_Spoof_normal.csv'
    filepath='./data/test.csv'
    
    # filepath='./data/data_test_6_DoS_150.csv'
    # filepath='./data/SPAT_2300.csv'
    train_size=0.8
    # 预处理
    data=preprocess(filepath,train_size)
    # data=preprocess_feature_filter(filepath,train_size)
    X_train, X_test, y_train, y_test=data

    dt,dt_train,dt_test=test1_Decisiontree(*data)
    rf,rf_train,rf_test=test2_Random_Forest(*data)
    et,et_train,et_test=test3_Extra_trees(*data)
    xg,xg_train,xg_test=test4_XGboost(*data)
    Stacking_4_model(dt_train, et_train, rf_train, xg_train,dt_test, et_test, rf_test, xg_test,y_train,y_test,X_test) 

    return
    # 特征选择
    data_after_fs=reprocess_feature_selection(filepath,train_size,dt,rf,et,xg)
    

    dt,dt_train,dt_test=test1_Decisiontree(*data_after_fs)
    rf,rf_train,rf_test=test2_Random_Forest(*data_after_fs)
    ef,ef_train,ef_test=test3_Extra_trees(*data_after_fs)
    xg,xg_train,xg_test=test4_XGboost(*data_after_fs)

if __name__=="__main__":
    main()