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

LABEL_COLUMN_NAME = 'death angle'
UNWANTED_COLUMNS = ['increase in deaths','current number of deaths day+1 per 100k','number of deaths per 100k','Tests (per 1k)','Prevalence of HIV total (% of population ages 15-49)','Death by Anemia female (per 100k)','Death by Anemia male (per 100k)','current number of deaths per 100k','number of deaths day+1 per 100k','number of cases per 100k','day','Death by Malaria male (per 100k)','Railways goods transported (normalized by population)','Smoking prevalence total (ages 15+)','Death by Malaria female (per 100k)','Death by Rheumatic Heart Disease male (per 100k)','world relative population','Death by Rheumatic Heart Disease female (per 100k)','number of flight passengers','Labor force female (% of total labor force)','People using safely managed drinking water services urban (% of urban population)','PM2.5 population exposed to levels exceeding WHO T-1 value (% of total)','PM2.5 population exposed to levels exceeding WHO T-2 value (% of total)','PM2.5 population exposed to levels exceeding WHO T-3 value (% of total)','Current health expenditure per capita (US$)','number of flight departures (normalized by population)','Death by Asthma male (per 100k)','Labor force participation rate total (% of total population ages 15+) (national estimate)','Mobile cellular subscriptions','Age 0-4','Age 5-9','Age 10-14','Age 15-19','Age 20-24','Age 25-29','Age 30-34','Age 35-39','Age 40-44','Age 45-49','Age 50-54','Age 55-59','Age 60-64','Age 65-69','Age 70-74','Age 75-79','Age 80-84','Age 85-89','Age 90-94','Age 95-99','Age 100+','HFC gas emissions (thousand metric tons of CO2 equivalent)','Urban area (sq. km)','Area (sq. km)','Mortality from CVD/Cancer/Diabetes ages 30-70 (%)','tuberculosis incidence (per 100k)','Mortality from CVD/Cancer/Diabetes ages 30-70 female (%)','Mortality from CVD/Cancer/Diabetes ages 30-70 male (%)','Agricultural nitrous oxide emissions (thousand metric tons of CO2 equivalent)','Air transport registered carrier departures worldwide','Labor force participation rate for ages 15-24 total (%) (national estimate)','Agricultural land (sq. km)','air pollution mean annual exposure (micrograms per cubic meter)','Rural land area (sq. km)','latitude','longitude','Labor force with advanced education male (% of male working-age population with advanced education)','Mortality rate attributed to air pollution (per 100k)','Mortality rate attributed to air pollution male (per 100k male population)','Mortality rate attributed to air pollution female (per 100k female population)','health expenditure','Labor force participation rate for ages 15-24 total (%) (national estimate).1','Nitrous oxide emissions (thousand metric tons of CO2 equivalent)','Agricultural methane emissions (thousand metric tons of CO2 equivalent)','Liner shipping connectivity index','Share of children covered by the DPT vaccine (%)','GDP per capita growth (annual %)','Air transport freight (million ton-km)','diabetes incidence','People using safely managed drinking water services (% of population)','People using safely managed drinking water services rural (% of rural population)','Forest area (% total)','Border size (Km)','Labor force total (% population)','Inflation GDP deflator (annual %)','People using at least basic sanitation services urban (% of urban population)','number of flight departures','International tourism receipts for travel items (current US$)','International tourism receipts (current US$)','number of deaths day-1 per 100k','number of deaths day-2 per 100k','Account ownership at a financial institution or with a mobile-money-service provider young adults (% of population ages 15-24)','population','Labor force participation rate female (% of female population ages 15-64) (modeled ILO estimate)','Labor force participation rate for ages 15-24 male (%) (national estimate)','GDP growth (annual %)','Access to electricity (% of population)','Access to electricity rural (% of rural population)','Access to electricity urban (% of urban population)','GDP','number of cases day-2 per 100k','number of cases day-1 per 100k','derivative number of deaths 1','derivative number of deaths 2','derivative number of cases 1','derivative number of cases 2','mortality rate','mortality rate day-1','mortality rate day-2','safe water index','Average hours of children study ages 7-14 (hours per week)','Average hours of children study female ages 7-14 (hours per week)','days of communitary spread','number of community workers per inhabitant','Risk of impoverishing expenditure for surgical care (% of people at risk)','Risk of catastrophic expenditure for surgical care (% of people at risk)','Account ownership at a financial institution or with a mobile-money-service provider older adults (% of population ages 25+)','Oil rents (% of GDP)','Coal rents (% of GDP)','Refugee population by country or territory of asylum','Refugee population by country or territory of origin','number of atms','availability of eletricity','number of cases day-2','number of cases day-1','log number of cases day-2','log number of cases day-1','number of deaths day-2','log number of deaths day-2','log number of deaths day-1','number of deaths day-1','number of deaths','number of cases','log number of cases','log number of deaths','name','number of deaths day+5','number of cases day+3','number of deaths day+1','number of deaths day+5 per 100k','mortality day+3','mortality day+5','mortality day+1','number of cases day+1 per 100k','number of cases day+3 per 100k','number of deaths day+3','number of cases day+5 per 100k','number of deaths day+3 per 100k','number of cases day+5','number of cases day+1','Death by Congenital Anomalies male (per 100k)','tuberculosis treatment success rate (%)','life expectancy at birth','International migrant stock (% of population)','air pollution population exposed to levels exceeding WHO guideline value (% of total)','Death by Stroke male (per 100k)','Death by Malnutrition female (per 100k)','Death by Inflammatory/Heart male (per 100k)','Death by Inflammatory/Heart female (per 100k)','Death by Malnutrition male (per 100k)','Death by Congenital Anomalies female (per 100k)','Death by Asthma female (per 100k)','Death by Stroke female (per 100k)','Suicide (deaths per 100k)','Conflict (deaths per 100k)','Suicide (deaths 5-14 years per 100k)','Homicide (deaths 5-14 years per 100k)','Suicide (deaths 15-49 years per 100k)','Homicide (deaths 15-49 years per 100k)','Suicide (deaths 50-69 years per 100k)','Suicide (deaths 70+ years per 100k)','Homicide (deaths 70+ years per 100k)','Neonatal disorders (deaths per 100k)','Natural disasters (deaths per 100k)','Natural disasters (deaths 70+ years per 100k)','Natural disasters (deaths 50-69 years per 100k)','Natural disasters (deaths 15-49 years per 100k)','Natural disasters (deaths 0-5 years per 100k)','Natural disasters (deaths 5-14 years per 100k)','Unsafe sex (deaths per 100k)','Fire (deaths 0-5 years per 100k)','Fire (deaths 5-14 years per 100k)','Fire (deaths 15-49 years per 100k)','Fire (deaths 50-69 years per 100k)','Fire (deaths 70+ years per 100k)','Road accidents (deaths 50-69 years per 100k)','Low birth weight (deaths per 100k)','Population female (% of total population)','Neonatal asphyxia & trauma (deaths 0-5 years per 100k)','Mathernal deaths per 100k births','Neonatal preterm complications (deaths 0-5 years per 100k)','Drowning (deaths 0-5 years per 100k)','Drowning (deaths 5-14 years per 100k)','Drowning (deaths 15-49 years per 100k)','Drowning (deaths 50-69 years per 100k)','Drowning (deaths 70+ years per 100k)','Homicide (deaths 0-5 years per 100k)','Heat-related deaths 70+ years per 100k (hot or cold exposure) (deaths 70+ years per 100k)','Heat (hot and cold exposure) (deaths per 100k)','Heat-related deaths 0-5 years per 100k (hot or cold exposure) (deaths 0-5 years per 100k)','Road accidents (deaths 70+ years per 100k)','Dementia (deaths 15-49 years per 100k)','Dementia (deaths 50-69 years per 100k)','Dementia (deaths 70+ years per 100k)','Dementia (deaths per 100k)','Road accidents (deaths 5-14 years per 100k)','Drowning (deaths per 100k)','Neonatal sepsis & infections (deaths 0-5 years per 100k)','Age 0-19','Age 20-59','Age 60-80','Age 80+','number of flight passengers (normalized by population)','Heat-related deaths 15-49 years per 100k (hot or cold exposure) (deaths 15-49 years per 100k)','Other neonatal disorders (deaths 0-5 years per 100k)','Low bone mineral density (deaths per 100k)','Non-exclusive breastfeeding (deaths per 100k)','Terrorism (deaths per 100k)','Diarrheal diseases (deaths 0-5 years per 100k)','Road accidents (deaths 15-49 years per 100k)','Maternal disorders (deaths per 100k)','Child wasting (deaths per 100k)','HIV/AIDS (deaths 0-5 years per 100k)','Malaria (deaths 0-5 years per 100k)','Kidney disease (deaths 0-5 years per 100k)','Homicide (deaths 50-69 years per 100k)','Investment in Vaccines','Discontinued breastfeeding (deaths per 100k)','Drug overdose (deaths 15-49 years per 100k)','Heat-related deaths 50-69 years per 100k (hot or cold exposure) (deaths 50-69 years per 100k)','Diarrheal diseases (deaths 50-69 years per 100k)','Diarrheal diseases (deaths 15-49 years per 100k)','Diarrheal diseases (deaths per 100k)','Diarrheal diseases (deaths 5-14 years per 100k)','Diarrheal diseases (deaths 70+ years per 100k)','Heat-related deaths 5-14 years per 100k (hot or cold exposure) (deaths 5-14 years per 100k)','Low physical activity (deaths per 100k)','Fiscal measures','High cholesterol (deaths per 100k)','Poisonings (deaths per 100k)','days after 100 cases','Road injuries (deaths per 100k)','Road accidents (deaths 0-5 years per 100k)','Diet low in legumes (deaths per 100k)','Meningitis (deaths 0-5 years per 100k)','Hepatitis (deaths 5-14 years per 100k)','Tuberculosis (deaths 0-5 years per 100k)','Intestinal infectious diseases (deaths per 100k)','Diabetes/blood/endocrine diseases (deaths 50-69 years per 100k)','Malaria (deaths 70+ years per 100k)','Digestive diseases (deaths 0-5 years per 100k)','Malaria (deaths 5-14 years per 100k)','Liver disease (deaths 0-5 years per 100k)','Hepatitis (deaths per 100k)','Human Development Index','High blood pressure (deaths per 100k)','Malaria (deaths 50-69 years per 100k)','Malaria (deaths per 100k)','Malaria (deaths 15-49 years per 100k)','Diet low in vegetables (deaths per 100k)','Disability-Adjusted Life Years (per 100k)','Parkinsons disease (deaths 15-49 years per 100k)','High blood sugar (deaths per 100k)','Air pollution (outdoor & indoor) (deaths per 100k)','Mortality due to air pollution (per 100k)','Cardiovascular diseases (deaths 0-5 years per 100k)','Cardiovascular diseases (deaths 5-14 years per 100k)','Cardiovascular diseases (deaths 15-49 years per 100k)','Diet low in seafood omega-3 fatty acids (deaths per 100k)','Protein-energy malnutrition (deaths 5-14 years per 100k)','Mobility trend (driving)','Mobility trend (walking)','Mobility trend (transit)','Protein-energy malnutrition (deaths 15-49 years per 100k)', 'Congenital birth defects (deaths 0-5 years per 100k)','Protein-energy malnutrition (deaths per 100k)','Protein-energy malnutrition (deaths 50-69 years per 100k)','Protein-energy malnutrition (deaths 70+ years per 100k)','Homicide (deaths per 100k)','Fire (deaths per 100k)','Diet high in red meat (deaths per 100k)','Diet low in whole grains (deaths per 100k)','Diet low in fruits (deaths per 100k)','Diet low in nuts and seeds (deaths per 100k)','Unsafe water source (deaths per 100k)','Whooping cough (deaths 0-5 years per 100k)','Indoor air pollution (deaths per 100k)','Cancers (deaths 0-5 years per 100k)','Diet high in sodium (deaths per 100k)','Kidney disease (deaths 5-14 years per 100k)','Digestive diseases (deaths 5-14 years per 100k)','Lower respiratory infections (deaths 5-14 years per 100k)','Liver disease (deaths 5-14 years per 100k)','Diabetes mellitus (deaths 5-14 years per 100k)','Lower respiratory infections (deaths 0-5 years per 100k)','Population male (% of total population)','Smoking prevalence males (% of adults)','Smoking prevalence females (% of adults)','Death by Flu/Pneumonia female (per 100k)','Monetary measures','Iron deficiency (deaths per 100k)','Liver diseases (deaths per 100k)','Drug use (deaths per 100k)','50 to 69 years old (share of deaths)','Child stunting (deaths per 100k)', '15 to 49 years old (share of deaths)','Diabetes mellitus (deaths 0-5 years per 100k)','Cardiovascular diseases (deaths per 100k)','Under 5s (share of deaths)','Diet low in fiber (deaths per 100k)','Testing framework (No testing policy)','5 to 14 years old (share of deaths)', 'Cancers (deaths per 100k)', 'Drug use disorders (deaths 70+ years per 100k)','Drug use disorders (deaths per 100k)','Zinc deficiency (deaths per 100k)', 'Drug disorder (deaths 50-69 years per 100k)','Contact tracing (Comprehensive contact tracing)','Secondhand smoke (deaths per 100k)','Air transport passengers carried (fraction of population)','Government health expenditure (% of general expenditure)','Cardiovascular diseases (deaths 50-69 years per 100k)','Current health expenditure (% of GDP)','Retail and recreation percent change from baseline (angle)','Grocery and pharmacy percent change from baseline (angle)','Parks percent change from baseline (angle)','Transit stations percent change from baseline (angle)','Workplaces percent change from baseline (angle)','Residential percent change from baseline (angle)','Kidney disease (deaths 50-69 years per 100k)','Testing framework (Open/Asymptomatic public testing)','Rural area (share of territory)','Parkinson disease (deaths 70+ years per 100k)','Emergency investment in health care','15-year olds (males per 100 females)','20-year olds (males per 100 females)','30-year olds (males per 100 females)','mean age','70-year olds (males per 100 females)','80-year olds (males per 100 females)','90-year olds (males per 100 females)','100-year olds (males per 100 females)','Tuberculosis (deaths 5-14 years per 100k)','HIV/AIDS (deaths 50-69 years per 100k)','HIV/AIDS (deaths 70+ years per 100k)','HIV/AIDS (deaths per 100k)','HIV/AIDS (deaths 5-14 years per 100k)','HIV/AIDS (deaths 15-49 years per 100k)', 'days after first case','Nutritional deficiencies (deaths 5-14 years per 100k)','Hepatitis (deaths 0-5 years per 100k)','Meningitis (deaths per 100k)','Population in the largest city (% urban population)','Kidney disease (deaths 15-49 years per 100k)', 'Death by Diabetes Mellitus female (per 100k)', 'Death by Cancer female (per 100k)','Poor sanitation (deaths per 100k)', 'Alcohol disorder (deaths 15-49 years per 100k)', 'Death by Hypertension male (per 100k)', 'Death by AIDS male (per 100k)','Death by Diabetes Mellitus male (per 100k)', 'Share of one person households (%)', '50-year olds (males per 100 females)', 'International tourism number of arrivals (per 100k)', 'Contact tracing (No contact tracing)', 'Kidney disease (deaths per 100k)','Population living in slums (% urban population)', 'No access to handwashing facility (deaths per 100k)', 'number of hospital beds (per 100k)','number of nurses (per 100k)','Population in urban agglomerations of more than 1 million (% of population)', 'number of physicians (per 100k)', 'Government health expenditure per capita PPP (US$)', 'Contact tracing (Limited contact tracing)', 'Individuals using the Internet (% of population)', 'Death by Flu/Pneumonia male (per 100k)', 'International travel controls (Quarantine on high-risk regions)', 'Digestive diseases (deaths per 100k)', 'Alcohol use disorders (deaths per 100k)', 'Tuberculosis (deaths per 100k)', 'Respiratory diseases (deaths per 100k)', 'Parkinson disease (deaths 50-69 years per 100k)', 'Liver disease (deaths 50-69 years per 100k)', 'Alcohol disorder (deaths 50-69 years per 100k)', 'Death by Tuberculosis female (per 100k)', 'Death by Tuberculosis male (per 100k)', 'Kidney disease (deaths 70+ years per 100k)', 'Liver disease (deaths 70+ years per 100k)', 'GDP per capita', 'Alcohol use (deaths per 100k)', 'Parkinsons disease (deaths per 100k)', 'Diabetes mellitus (deaths 15-49 years per 100k)']
#, 'Testing framework (Symptoms or specific criteria)'


def eval_cv(df1, df2, df3, features):
    X_train = df1[features].values
    y_train = df1[LABEL_COLUMN_NAME].values
    X_val = df2[features].values
    y_val = df2[LABEL_COLUMN_NAME].values
    X_test = df3[features].values
    y_test = df3[LABEL_COLUMN_NAME].values

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
    #gbm = lgb.train(params, lgb_train, num_boost_round=100, verbose_eval=False)
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


df1 = pd.read_csv(sys.argv[1])
df2 = pd.read_csv(sys.argv[2])
df3 = pd.read_csv(sys.argv[3])
df1.dropna(axis=0, subset=[LABEL_COLUMN_NAME], inplace=True)
df2.dropna(axis=0, subset=[LABEL_COLUMN_NAME], inplace=True)
df3.dropna(axis=0, subset=[LABEL_COLUMN_NAME], inplace=True)

all_features = list(df1.columns)
for f in UNWANTED_COLUMNS + [LABEL_COLUMN_NAME]:
    all_features.remove(f)

#f = ['Average Temperature (C)', '40-year olds (males per 100 females)', 'Income Distribution (GINI Index)', '65+ years (share of population)', 'Restrictions on internal movement (No measures)', 'Out-of-pocket expenditure (% health expenditure)', 'Diabetes mellitus (deaths 70+ years per 100k)', 'Nutritional deficiencies (deaths per 100k)', '70+ years old (share of deaths)', '25 to 64 years (share of population)', 'Vitamin-A deficiency (deaths per 100k)', '15 to 24 years (share of population)', 'Public information campaigns (No campaign)', 'Testing framework (Symptoms or specific criteria)', 'Digestive diseases (deaths 15-49 years per 100k)', 'Under 5s (share of population)', 'Liver disease (deaths 15-49 years per 100k)', 'Nutritional deficiencies (deaths 0-5 years per 100k)', 'Hepatitis (deaths 15-49 years per 100k)', '60-year olds (males per 100 females)', 'Nutritional deficiencies (deaths 15-49 years per 100k)', 'Lower respiratory infections (deaths 15-49 years per 100k)', 'Obesity (deaths per 100k)', 'Cancers (deaths 70+ years per 100k)']
f = ['Average Temperature (C)', '40-year olds (males per 100 females)', 'Income Distribution (GINI Index)', '65+ years (share of population)', 'Restrictions on internal movement (No measures)', 'Out-of-pocket expenditure (% health expenditure)', 'Diabetes mellitus (deaths 70+ years per 100k)', 'Nutritional deficiencies (deaths per 100k)', '70+ years old (share of deaths)', '25 to 64 years (share of population)', 'Vitamin-A deficiency (deaths per 100k)', '15 to 24 years (share of population)', 'Public information campaigns (No campaign)']
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
    #if i > 2 and xx.count(f) == 0:
    #     xx.insert(len(xx), f)
    #     v,ft = back_one(df1, df2, df3, f)
    #     if v == 1: f = ft
    #     while i > 0 and v == 1:
    #         v,f = back_one(df1, df2, df3, f)
    #     i = len(f)
