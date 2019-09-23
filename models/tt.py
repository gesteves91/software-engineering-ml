import pandas as pd
import numpy as np
import sys
import os
import random

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from statsmodels import robust
from scipy import stats

from sklearn.model_selection import KFold
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso

#duvida SLC15, ASRI42, ASRI_pass42 (pensava em morrer ou em suicidio)

# Parameters
LABEL_COLUMN_NAME = 'mean_commits'
UNWANTED_COLUMNS = []
#UNWANTED_COLUMNS = ['ASRI54','ASRI42','ASRI_pass42','ASRI_pass54','num_tent_suicidio','metodo_suicidio_0','metodo_suicidio_1','metodo_suicidio_2','metodo_suicidio_3','metodo_suicidio_4','metodo_suicidio_5','metodo_suicidio_6','metodo_suicidio_7','emerg_suicidio','suic_impulso','sentim_suicid_0','sentim_suicid_1','sentim_suicid_2','sentim_suicid_3','sentim_suicid_4','sentim_suicid_5','sentim_suicid_6','xxx440','xxx442','xxx444','xxx446','xxx448','xxx452','xxx456','xxx458','xxx460','xxx462','xxx464','xxx466','xxx468','xxx470','xxx472','xxx473','xxx474','xxx475','xxx476','xxx478','xxx479','xxx480','xxx481','xxx482','xxx483','xxx484','xxx485','xxx486','xxx487','xxx488','xxx489','xxx490','xxx494','xxx495','xxx496','xxx497','xxx498','xxx499','xxx500','xxx501','xxx502','xxx503','xxx505','xxx506','xxx508','xxx509','xxx510','xxx511','xxx512','xxx513','xxx514','xxx515','xxx516','xxx518','xxx519','xxx520','xxx521','xxx522','xxx523','xxx524','xxx525','xxx526','xxx527','xxx528','xxx529','xxx530','xxx531','xxx532','xxx533','xxx534','xxx535','xxx536','xxx537','xxx538','xxx540','xxx542','xxx825','cadastro1','cadastro2','cadastro3','cadastro4','cadastro5','cadastro6','cadastro7','cadastro8','cadastro9','cadastro10','cadastro11','cadastro12','cadastro13','cadastro14','cadastro15','cadastro16','cadastro17']

N_FOLDS = 5
RANDOM_STATE = 1

def eval_bootstrap(df, features, md):
    X = df[features].values
    y = df[LABEL_COLUMN_NAME].values

    aa = []
    bb = []
    cc = []
    dd = []
    for i in range(1,5):
        a = []
        b = []
        c = []
        d = []
        cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=i)
        for (train, val) in cv.split(X, y):
            if md == 1: regressor = ensemble.GradientBoostingRegressor(n_estimators = 30, max_depth = 4, min_samples_split = 2, learning_rate = 0.1, loss = 'ls', random_state = RANDOM_STATE)
            elif md == 2: regressor = ensemble.RandomForestRegressor(n_estimators = 30, max_depth = 10, min_samples_split = 4, random_state = RANDOM_STATE)
            elif md == 3: regressor = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
            elif md == 4: regressor = MLPRegressor(hidden_layer_sizes=(20,30,30,5,), batch_size = 10, activation='relu', random_state = RANDOM_STATE)
            elif md == 5: regressor = LinearRegression()
            elif md == 6: regressor = Lasso(alpha=0.1, random_state = RANDOM_STATE)

            regressor = regressor.fit(X[train], y[train])
            pred = regressor.predict(X[val])

            rmse = np.sqrt(np.mean((pred - y[val])**2))
            mae = mean_absolute_error(pred, y[val])
            r2 = r2_score(pred, y[val])

            a.insert(len(a), rmse)
            b.insert(len(b), mae)
            c.insert(len(c), r2)

        aa.append(np.mean(a))
        bb.append(np.mean(b))
        cc.append(np.mean(c))
    return np.mean(aa),np.mean(bb),np.mean(cc)

def back_one(df, f, md):
    v = 0
    f1 = []
    f2 = []
    for i in f:
        f1.insert(len(f1), i)
        f2.insert(len(f2), i)
    A,B,C = eval_bootstrap(df, f1, md)
    z = A
    for i in f:
        f1.remove(i)
        A,B,C = eval_bootstrap(df, f1, md)
        print("%s,%f,%f,%f" % (f1,A,B,C))
        if A < z:
            v = 1
            z = A
            f2 = []
            for j in f1:
                f2.insert(len(f2), j)
        f1.insert(len(f1), i)
    return v,f2

# Reads dataset
df = pd.read_csv(sys.argv[1])
df.dropna(axis=0, subset=[LABEL_COLUMN_NAME], inplace=True)

RANDOM_STATE = 1
all_features = list(df.columns)

#f = []
#for x in UNWANTED_COLUMNS:
#    if x in all_features: f.insert(len(f), x)
#for x in f + [LABEL_COLUMN_NAME]:
for x in UNWANTED_COLUMNS + [LABEL_COLUMN_NAME]:
    all_features.remove(x)

md = int(sys.argv[2])
f = []
i = 0
for f1 in all_features:
    if i == 5: break
    if f1 in f: continue
    k = 1000
    x = f1
    i = i + 1
    j = 0
    for f2 in all_features:
         if f2 in f: continue
         j = j + 1
         f.insert(len(f), f2)
         A,B,C = eval_bootstrap(df, f, md)
         print("%s,%f,%f,%f" % (f,A,B,C))
         z = A
         f.remove(f2)
         sys.stdout.flush()
         if z < k:
             x = f2
             k = z
    f.insert(len(f), x)
    if i > 2:
         v,f = back_one(df, f, md)
         while v == 1:
             v,f = back_one(df, f, md)
         i = len(f)
