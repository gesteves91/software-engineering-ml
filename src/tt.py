import pandas as pd
import numpy as np
import matplotlib
import sys
import os
import random

import lightgbm as lgb

from sklearn.metrics import mean_absolute_error

params = {
    "max_bin": 64,
    "max_depth": 20,
    "learning_rate": 0.1,
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": 'mae',
    "num_leaves": 35,
    "verbose": -1,
    "min_data": 10,
    "boost_from_average": False,
    'bagging_freq': 1,
    "random_state": 1
}

LABEL_COLUMN_NAME = 'mean_commits'
UNWANTED_COLUMNS = ['comment_count', 'total_deletions', 'total_additions', 'total',
                    'private', 'fork', 'forks', 'open_issues', 'watchers', 'network_count', 
                    'admin', 'push',
                    'pull',  'language_C#', 'language_C++', 'language_CSS', 'language_CoffeeScript',
                    'language_Go', 'language_Perl',
                    'language_Shell', 'language_TypeScript',
                    'most_changes_added',
                    'most_changes_modified', 'most_changes_removed', 'most_changes_renamed',
                    'most_changes_unknown', 'is_night',
                    'is_weekend',
                    'type_User','date'
                    ]

def eval_cv(df1, df2, df3, features):
    X_train = df1[features].values
    y_train = df1[LABEL_COLUMN_NAME].values
    X_val = df2[features].values
    y_val = df2[LABEL_COLUMN_NAME].values
    X_test = df3[features].values
    y_test = df3[LABEL_COLUMN_NAME].values

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

    gbm = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=lgb_eval, early_stopping_rounds=10, verbose_eval=False)
    y_pred = gbm.predict(X_val)
    y_last = gbm.predict(X_test)

    for i in range(len(y_val)):
         #if y_pred[i] < 0: y_pred[i] = 0
         print("result: ",i,y_val[i],y_pred[i])
    print("result: ",i+1,y_test[0],y_last[0])

    return mean_absolute_error(y_val, y_pred)

def back_one(df1, df2, df3, f):
    v = 0
    f1 = []
    f2 = []
    for i in f:
        f1.insert(len(f1), i)
        f2.insert(len(f2), i)
    A = eval_cv(df1, df2, df3, f1)
    z = A
    for i in f:
        f1.remove(i)
        A = eval_cv(df1, df2, df3, f1)
        print("%s,%f" % (f1,A))
        if A < z:
            v = 1
            z = A
            f2 = []
            for j in f1:
                f2.insert(len(f2), j)
        f1.insert(len(f1), i)
    return v,f2


#df1 = pd.read_csv(sys.argv[1])
df1 = pd.read_csv('../experiments/scala.csv')

def separate_data(df):
    train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
    return validate, test

df2, df3 = separate_data(df1)

#df2 = pd.read_csv(sys.argv[2])
#df3 = pd.read_csv(sys.argv[3])
df1.dropna(axis=0, subset=[LABEL_COLUMN_NAME], inplace=True)
df2.dropna(axis=0, subset=[LABEL_COLUMN_NAME], inplace=True)
df3.dropna(axis=0, subset=[LABEL_COLUMN_NAME], inplace=True)

all_features = list(df1.columns)
for f in UNWANTED_COLUMNS + [LABEL_COLUMN_NAME]:
    all_features.remove(f)

#f = ['Average Temperature (C)', '40-year olds (males per 100 females)', 'Income Distribution (GINI Index)', '65+ years (share of population)', 'Restrictions on internal movement (No measures)', 'Out-of-pocket expenditure (% health expenditure)', 'Diabetes mellitus (deaths 70+ years per 100k)', 'Nutritional deficiencies (deaths per 100k)', '70+ years old (share of deaths)', '25 to 64 years (share of population)', 'Vitamin-A deficiency (deaths per 100k)', '15 to 24 years (share of population)', 'Public information campaigns (No campaign)']
f = ['phases_project', 'language_C', 'language_Python', 'language_R', 'language_Ruby', 'language_Java',
                  'has_wiki', 'has_downloads', 'outside_contribution', 'tests_included',
                  'language_PHP', 'type_Organization', 'has_issues',  'language_Scala', 'size',
                  'total_files',
                  'language_JavaScript',

                  ]
xx = []
i = 0

for f1 in all_features:
    if i == 50: break
    if f1 in f: continue
    k = 10000
    x = f1
    i = i + 1
    j = 0
    for f2 in all_features:
         if f2 in f: continue
         j = j + 1
         f.insert(len(f), f2)
         A = eval_cv(df1, df2, df3, f)
         print("%s,%f" % (f,A))
         z = A
         f.remove(f2)
         sys.stdout.flush()
         if z < k:
             x = f2
             k = z
    f.insert(len(f), x)
