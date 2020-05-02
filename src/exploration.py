import pandas as pd
import numpy as np
import sys
import os
import random

from itertools import combinations
from sklearn.model_selection import KFold
from statsmodels import robust
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score

from xgboost import XGBRegressor
from xgboost import XGBClassifier

from itertools import cycle
from scipy import interp

#file_features = open("features.txt", "w+")
#file_auc = open("auc_all.txt", "w+")

# Parameters
LABEL_COLUMN_NAME = 'mean_commits'
UNWANTED_COLUMNS = ['comment_count', 'total_deletions', 'total_additions', 'total',
                    'private', 'fork',
                    'forks', 'open_issues', 'watchers', 'network_count', 'admin', 'push',
                    'pull',  'language_C#', 'language_C++', 'language_CSS', 'language_CoffeeScript',
                    'language_Go', 'language_Perl',

                    'language_Shell', 'language_TypeScript',
                    'most_changes_added',
                    'most_changes_modified', 'most_changes_removed', 'most_changes_renamed',
                    'most_changes_unknown',
                    'total_files',
                    'language_JavaScript',
                    'is_night',
                    'is_weekend',
                    'type_User', ]

WANTED_COLUMNS = ['phases_project', 'language_C', 'language_Python', 'language_R', 'language_Ruby', 'language_Java',
                  'has_wiki', 'has_downloads', 'outside_contribution', 'tests_included',
                  'language_PHP', 'type_Organization', 'has_issues',  'language_Scala', 'size'
                  ]

N_FOLDS = 2
RANDOM_STATE = 1

n_estimators = 20
subsample = 0.6
lr = 0.1
max_depth = 10

total = 0
best_models = 0
best_generated_model = 0
feat = []

for c in range(1, 50):
    feat.append('feature')


def random_combinations(iterable, r, x):
    pool = tuple(iterable)
    n = len(pool)
    a = []
    for i in range(x):
        indices = sorted(random.sample(range(n), r))
        a.insert(len(a), tuple(pool[i] for i in indices))
    return list(set(a))


def eval_features(df, features):
    global total
    global best_models

    total = total + 1

    X = df[features].values
    y = df[LABEL_COLUMN_NAME].values
    a = []
    b = []
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=i)
    for (train, val) in cv.split(X, y):
        regressor = XGBRegressor(n_estimators=n_estimators, subsample=subsample,
                                 learning_rate=lr, max_depth=max_depth, n_jobs=32,
                                 random_state=1, objective='reg:squarederror')

        regressor = regressor.fit(X[train], y[train])
        pred = regressor.predict(X[val])

        r2 = r2_score(pred, y[val])

    return r2


def eval_panel(df, comb):
    for ff in comb:
        f = []
        for x in ff:
            f.insert(len(f), x)
        A = eval_features(df, f)
        #print("%s,%f,%s,%s" % (f, np.mean(A),A,B))
        check_best_models(A, f)
        print("%s,%f" % (f, A))
        #file_features.write(str(f) + "\n")
        #file_auc.write(str(A) + "\n")
        sys.stdout.flush()


def check_best_models(acc, features):
    global best_models, best_generated_model, feat

    model_accuracy = acc

    # check the number of models above the baseline model
    if (model_accuracy > 0.75):
        best_models = best_models + 1
        if (len(features) < len(feat)):
            feat = features
    # check the highestes model achieved
    if (model_accuracy > best_generated_model):
        best_generated_model = model_accuracy


# Reads dataset
df_mblood = pd.read_csv(sys.argv[1])

# Maps label
df_mblood.dropna(axis=0, subset=['mean_commits'], inplace=True)

RANDOM_STATE = 1
f = []
i = 0
# for f1 in all_features:
for f1 in WANTED_COLUMNS:
    if i == 15:
        break
    if f1 in f:
        continue
    k = 0
    x = f1
    i = i + 1
    j = 0
    avg = 0
#    for f2 in all_features:
    for f2 in WANTED_COLUMNS:
        if f2 in f:
            continue
        j = j + 1
        f.insert(len(f), f2)
        A = eval_features(df_mblood, f)
        #print("%s,%f,%s,%s" % (f,np.mean(A),A,B))
        check_best_models(A, f)
        print("%s,%f" % (f, A))
        #file_features.write(str(f) + "\n")
        #file_auc.write(str(A) + "\n")
        f.remove(f2)
        sys.stdout.flush()
        avg = avg + np.mean(A)
        if np.mean(A) > k:
            x = f2
            k = np.mean(A)
    avg /= j
    f.insert(len(f), x)

for c in range(1, 5):
    s = 50000
    comb = random_combinations(WANTED_COLUMNS, c, s)
    eval_panel(df_mblood, comb)

percentage = (best_models / total) * 100

with open('../reports/total.txt', 'w') as f:
    print("Total number of models: %i\nBest achieved model: %f\nFeatures related to the smallest set of features: %s\nNumber of best models: %i \nPercentage of best models: %f" % (
        total, best_generated_model, feat, best_models, percentage), file=f)

# file_features.close()
# file_auc.close()

print('file written!!!')
