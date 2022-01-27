import warnings
warnings.filterwarnings('ignore')

import acv_explainers as aa
from acv_explainers import ACVTree
# import shap

import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBRFClassifier, XGBClassifier, XGBRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import random
import time
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import comb
from experiments.utils import *

random.seed(2021)
np.random.seed(2021)

from experiments.exp_syn import *

# Fixed the seed and plot env
random.seed(1)
np.random.seed(1)

plt.style.use(['ggplot'])

## Create synthetic dataset and train a RandomForest

p = 0.8
n = 10000
d = 20
C = [[]]
# mean
mean = np.zeros(d)

# Determinitist covariance
cov = p*np.ones(shape=(d, d)) + 20*np.eye(d)

# b = np.random.randn(d, d)
# cov = np.dot(b, b.T)

model_type = 'syn7'

coefs = 4*np.random.randn(d)
coefs[0] = 2
coefs[1] = 4

coefs[2] = -4
coefs[3] = -3

exp = ExperimentsLinear(mean=mean, cov=cov, n=n, C=C, data_type=model_type, coefs=coefs)
model = RandomForestRegressor(n_estimators=10, max_depth=6)
model.fit(exp.data, exp.y_train)

acvtree = ACVTree(model, exp.data)


S = np.array(list(range(d)))
print(acvtree.compute_quantile_shaff(X=exp.data[:5], S=S, data=exp.data[:10000], y_data=exp.y_train[:10000], quantile=20).reshape(-1))
print(mc_cond_exp(X=exp.data[:5], tree=acvtree, S=S, mean=mean, cov=cov, N=10000).reshape(-1))
print(acvtree.compute_exp_shaff(X=exp.data[:5], S=S, data=exp.data[:10000], y_data=exp.y_train[:10000]).reshape(-1))
print(acvtree.compute_exp_normalized(X=exp.data[:5], S=S, data=exp.data[:10000]).reshape(-1))
print(shap_cond_exp(X=exp.data[:5], tree=acvtree, S=S).reshape(-1))
# print(acvtree.predict(exp.data[0]))