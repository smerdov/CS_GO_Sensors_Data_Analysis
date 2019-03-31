import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from utils import normalize_MPU9250_data, split_df
from sklearn.metrics import mutual_info_score
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, LassoCV, LassoLarsIC
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import argparse


plt.interactive(True)
pd.options.display.max_columns = 15
pic_prefix = 'pic/'

parser = argparse.ArgumentParser()
parser.add_argument('--suffix', default='', type=str)
args = parser.parse_args()
suffix = args.suffix

ss = StandardScaler()

# median_cols = ['median_acc_x_std_2s', 'median_acc_y_std_2s', 'median_acc_z_std_2s',
#        'median_gyro_x_std_2s', 'median_gyro_y_std_2s', 'median_gyro_z_std_2s']

# target_nums_list = [100, 500, 3000]
target_nums_list = [1000]
target_num = 1000

for target_num in target_nums_list:

    df_features = pd.read_csv('data/df_features.csv')
    df_features.drop(['session_id', 'player_id'], axis=1, inplace=True)
    # target = df_features.loc[:, '>500 h']
    target = df_features.loc[:, f'>{target_num} h exp']
    df_train = df_features.iloc[:, 4:]  ### WARNING: MANUAL ASSIGNMENT
    # median_cols = df_train.columns[6:12]
    # df_train.drop(median_cols, axis=1, inplace=True)
    df_train = df_train.fillna(df_train.median())
    df_train.loc[:, :] = ss.fit_transform(df_train)
    target_normalized = ss.fit_transform(target.values.reshape(-1, 1))

    lasso = Lasso(alpha=0.07, normalize=False)
    lasso.fit(df_train, target_normalized)
    lasso = LinearSVC(penalty='l1', loss='l2', dual=False, C=0.05)
    lasso.fit(df_train, target)

    df_coef = pd.DataFrame(lasso.coef_.ravel(), index=df_train.columns, columns=['coef']).sort_values('coef')
    df_coef = df_coef.loc[df_coef['coef'] != 0].round(2)
    df_coef.to_csv(f'data/df_coef_{target_num}.csv', index=True)

    model_aic = LassoLarsIC(criterion='aic', max_iter=10000, normalize=False)
    # model_aic.fit(df_train, target)
    model_aic.fit(df_train, target_normalized)
    # model_aic.coef_
    # model_aic.alpha_
    # model_aic.alphas_

    model_bic = LassoLarsIC(criterion='bic', max_iter=10000, normalize=False)
    # model_bic.fit(df_train, target)
    model_bic.fit(df_train, target_normalized)

# model_bic.criterion_
# model_bic.alpha_
# model_bic.coef_

def plot_ic_criterion(model, name, color):
    alpha_ = model.alpha_
    alphas_ = model.alphas_
    mask = alphas_ > 0
    n_features = len(np.nonzero(model.coef_)[0])

    criterion_ = model.criterion_
    plt.plot(-np.log10(alphas_[mask]), criterion_[mask], color=color,
             linewidth=3, label=f'{name} criterion, {n_features} features')
    plt.axvline(-np.log10(alpha_), linestyle='--', color=color, linewidth=2,
                label=r'$\alpha$: %s estimate' % name)
    plt.xlabel(r'-log($\alpha$)')
    plt.ylabel('Criterion value')

plt.close()
plt.figure()
plot_ic_criterion(model_aic, 'AIC', 'b')
plot_ic_criterion(model_bic, 'BIC', 'r')
plt.legend()
plt.title('Information criteria for model selection')
plt.tight_layout(rect=[-0.01, -0.02, 1.01, 1.02])
plt.savefig(f'{pic_prefix}information_criteria_{suffix}.png')


df_coef_aic = pd.DataFrame(model_aic.coef_.ravel(), index=df_train.columns, columns=['coef']).sort_values('coef')
df_coef_aic = df_coef_aic.loc[df_coef_aic['coef'] != 0].round(2)
df_coef_aic.to_csv(f'data/df_coef_aic_{suffix}.csv')

df_coef_bic = pd.DataFrame(model_bic.coef_.ravel(), index=df_train.columns, columns=['coef']).sort_values('coef')
df_coef_bic = df_coef_bic.loc[df_coef_bic['coef'] != 0].round(2)
df_coef_bic.to_csv(f'data/df_coef_bic_{suffix}.csv')





# from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
#
# mutual_information_list = []
#
# for _ in range(50):
#     mutual_information = mutual_info_classif(df_train, target, discrete_features=False)
#     df_mutual_info = pd.DataFrame(mutual_information, index=df_train.columns, columns=['mut_info'])
#     mutual_information_list.append(df_mutual_info)
#     # mutual_information_list.append(df_mutual_info.sort_values(['mut_info']))
#
# mut_info_mean = pd.concat(mutual_information_list, axis=1).mean(axis=1)
# mut_info_mean.sort_values()







