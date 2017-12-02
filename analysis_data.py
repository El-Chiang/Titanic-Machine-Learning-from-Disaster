#coding=utf-8
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文标签字体

data_train = pd.read_csv(r"./data/train.csv")  # 读取csv
# print(data_train)
# print(data_train.info())
# print(data_train.describe())

def first_analysis():
    fig = plt.figure()
    fig.set(alpha=0.2)

    plt.subplot2grid((2, 3), (0, 0))
    data_train.Survived.value_counts().plot(kind="bar")  # 柱状图
    plt.title('获救情况')
    plt.ylabel('人数')

    plt.subplot2grid((2, 3), (0, 1))
    data_train.Pclass.value_counts().plot(kind="bar")
    plt.ylabel('人数')
    plt.title('乘客等级分布')

    plt.subplot2grid((2, 3), (0, 2))
    plt.scatter(data_train.Survived, data_train.Age)
    plt.ylabel('年龄')
    plt.grid(b=True, which='major', axis='y')
    plt.title('通过年龄看获救分布(1为获救)')

    plt.subplot2grid((2, 3), (1, 0), colspan=2)
    data_train.Age[data_train.Pclass == 1].plot(kind='kde')
    data_train.Age[data_train.Pclass == 2].plot(kind='kde')
    data_train.Age[data_train.Pclass == 3].plot(kind='kde')
    plt.xlabel('年龄')
    plt.ylabel('密度')
    plt.title('各等级的乘客年龄分布')
    plt.legend(('头等舱', '二等舱', '三等舱'), loc='best')

    plt.subplot2grid((2, 3), (1, 2))
    data_train.Embarked.value_counts().plot(kind='bar')
    plt.title('各登船港口上船人数')
    plt.ylabel('人数')
    plt.show()

# 属性与获救结果的关联统计
def pclass_survived():
    """各乘客等级的获救情况
    """
    fig = plt.figure()
    fig.set(alpha=0.2)

    Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
    Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
    df = pd.DataFrame({'获救': Survived_1, '未获救': Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.title('各乘客等级的获救情况')
    plt.xlabel('乘客等级')
    plt.ylabel('人数')
    plt.show()

def sex_survived():
    """各性别的获救情况
    """
    fig = plt.figure()
    fig.set(alpha=0.2)

    Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
    Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
    df=pd.DataFrame({u'男性':Survived_m, u'女性':Survived_f})
    df.plot(kind='bar', stacked=True)
    plt.title("按性别看获救情况")
    plt.xlabel("性别") 
    plt.ylabel("人数")
    plt.show()

def sex_pclass_survived():
    """各种舱级别情况下性别的获救情况
    """
    fig = plt.figure()
    fig.set(alpha=0.2)

    plt.title('根据舱等级和性别的获救情况')
    ax1 = fig.add_subplot(141)
    data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='female, high-class', color='deeppink')
    ax1.set_xticklabels(['获救', '未获救'], rotation=0)
    ax1.legend(['女性/高级舱'], loc='best')

    ax2 = fig.add_subplot(142, sharey=ax1)
    data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low-class', color='pink')
    ax2.set_xticklabels(['未获救', '获救'], rotation=0)
    plt.legend(['女性/低级舱'], loc='best')

    ax3 = fig.add_subplot(143, sharey=ax1)
    data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high-class', color='lightblue')
    ax3.set_xticklabels(['未获救', '获救'], rotation=0)
    plt.legend(['男性/高级舱'], loc='best')
    
    ax4 = fig.add_subplot(144, sharey=ax1)
    data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male, low-class', color='steelblue')
    ax4.set_xticklabels(['未获救', '获救'], rotation=0)
    plt.legend(['男性/低级舱'], loc='best')
    
    plt.show()

def embarked_survived():
    """各登船港口的获救情况
    """
    fig = plt.figure()
    fig.set(alpha=0.2)

    Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
    Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
    df = pd.DataFrame({'获救': Survived_1, '未获救': Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.title('各登船港口乘客的获救情况')
    plt.xlabel('登船港口')
    plt.ylabel('人数')
    plt.show()

def sibsp_survived():
    """堂兄弟/妹，孩子/父母有几人对是否获救的影响
    """
    g = data_train.groupby(['SibSp', 'Survived'])
    df = pd.DataFrame(g.count()['PassengerId'])
    print(df)

    g = data_train.groupby(['Parch', 'Survived'])
    df = pd.DataFrame(g.count()['PassengerId'])
    print(df)

def if_cabin_survived():
    fig = plt.figure()
    fig.set(alpha=0.2)

    Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
    Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
    df = pd.DataFrame({'有': Survived_cabin, '无': Survived_nocabin}).transpose()
    df.plot(kind='bar', stacked=True)
    plt.title('根据Carin有无看获救情况')
    plt.xlabel('Cabin有无')
    plt.ylabel('人数')
    plt.show()

if __name__ == '__main__':
    first_analysis()
    pclass_survived()
    sex_survived()
    sex_pclass_survived()
    embarked_survived()
    sibsp_survived()
    if_cabin_survived()