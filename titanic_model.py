import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import cross_validation, linear_model, preprocessing
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.learning_curve import learning_curve

data_train = pd.read_csv(r"./data/train.csv")  # 读取csv

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文标签字体


def set_missing_ages(df):
    """使用RandomForestClassifier填补缺失的年龄
    """
    # 把已有的数值型特征取出来丢进RandomForestRegressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predicteAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predicteAges

    return df, rfr


def set_cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = 'No'
    return df


# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """画出data在某模型上的learning curve
    ------------------
    estimator: 分类器
    title: 表格的标题
    X: 输入的feature, numpy类型
    y: 输入的target vector
    ylim: tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv: 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training
    n_jobs: 并行的任务数
    """
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes, verbose=verbose)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean,
                 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean,
                 'o-', color="r", label=u"交叉验证集上得分")
        plt.legend(loc="best")
        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) +
                (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]
            ) - (test_scoreds_mean[-1] - test_scores_std[-1])
    return midpoint, diff

if __name__ == '__main__':
    data_train, rfr = set_missing_ages(data_train)
    data_train = set_cabin_type(data_train)

    # 特征因子化
    dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')

    df = pd.concat([data_train, dummies_Cabin, dummies_Embarked,
                    dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket',
            'Cabin', 'Embarked'], axis=1, inplace=True)

    # 对Age和Fare属性进行scaling
    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))
    df['Age_scaled'] = scaler.fit_transform(
        df['Age'].values.reshape(-1, 1), age_scale_param)
    fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))
    df['Fare_scaled'] = scaler.fit_transform(
        df['Fare'].values.reshape(-1, 1), fare_scale_param)

    # 逻辑回归建模

    # 用正则取出属性值
    train_df = df.filter(
        regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.as_matrix()

    # y即Survival结果
    y = train_np[:, 0]

    # X即特征属性值
    X = train_np[:, 1:]

    # fit到RandomForestRegressor中
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X, y)

    # 对test.csv做数据预处理
    data_test = pd.read_csv(r"./data/test.csv")
    data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
    # 特征变换
    # 先用RandomForestRegressor填补丢失的年龄
    tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    null_age = tmp_df[data_test.Age.isnull()].as_matrix()
    # 根据特征属性X预测年龄并补上
    X = null_age[:, 1:]
    predictedAges = rfr.predict(X)
    data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges

    data_test = set_cabin_type(data_test)
    dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')

    df_test = pd.concat([data_test, dummies_Cabin,
                        dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket',
                'Cabin', 'Embarked'], axis=1, inplace=True)
    df_test['Age_scaled'] = scaler.fit_transform(
        df_test['Age'].values.reshape(-1, 1), age_scale_param)
    df_test['Fare_scaled'] = scaler.fit_transform(
        df_test['Fare'].values.reshape(-1, 1), fare_scale_param)

    # 获取预测结果
    test = df_test.filter(
        regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = clf.predict(test)
    result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(
    ), 'Survived': predictions.astype(np.int32)})
    # result.to_csv('./data/logistic_regression_predictions.csv', index=False)

    # 交叉验证
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    all_data = df.filter(
        regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    X = all_data.as_matrix()[:, 1:]
    y = all_data.as_matrix()[:, 0]
    # print(cross_validation.cross_val_score(clf, X, y, cv=5))

    # 分割数据，按照训练数据:cv数据=7:3
    split_train, split_cv = cross_validation.train_test_split(
        df, test_size=0.3, random_state=0)
    train_df = split_train.filter(
        regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    # 生成模型
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(train_df.as_matrix()[:, 1:], train_df.as_matrix()[:, 0])

    # 对cross validacation数据进行验证
    cv_df = split_cv.filter(
        regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = clf.predict(cv_df.as_matrix()[:, 1:])

    origin_data_train = pd.read_csv('./data/train.csv')
    bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(
        split_cv[predictions != cv_df.as_matrix()[:, 0]]['PassengerId'].values)]
    # print(bad_cases)

    # plot_learning_curve(clf, u"学习曲线", X, y) # learning curve

    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
    train_np = train_df.as_matrix()

    # y即Survival结果
    y = train_np[:, 0]
    
    # X即特征属性值
    X = train_np[:, 1:]

    # fit到BaggingRegressor
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
    bagging_clf.fit(X, y)

    test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
    predictions = bagging_clf.predict(test)
    result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
    result.to_csv('./data/logistic_regression_bagging_predictions.csv', index=False)