import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from utils import string2json
from config import TIMESTEP
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score



plt.interactive(True)
pd.options.display.max_columns = 15
pic_folder = 'pic/'

target_column = 'kills_proportion'
df4train = pd.read_csv('data/df4train.csv')
target = df4train[target_column]
target_binary = (target > target.median()) * 1
df4train.drop(columns=['timedelta', target_column], inplace=True)
# df4train.set_index('timedelta', inplace=True)



ss = StandardScaler()
df4train.loc[:, :] = ss.fit_transform(df4train.values)

# lr = LinearRegression()
lr = LogisticRegression()

lr.fit(df4train, target_binary)

predict = lr.predict(df4train)
roc_auc_score(target_binary, predict)
accuracy_score(target_binary, predict)


mean_squared_error(target, predict)
mean_squared_error(target, np.repeat(target.mean(), len(target)))


pd.Series(data=lr.coef_[0], index=df4train.columns)



net = nn.Rec























