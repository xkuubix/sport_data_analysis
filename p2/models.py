import sys
import os
sys.path.append(os.path.abspath('../')) 
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
import utils
from sklearn.linear_model import LinearRegression, HuberRegressor
import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# %%
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_theme()
DATA_PATH = "/media/dysk_a/jr_buler/Healthy_Sport"
FILE_NAME = "utf-8_RESEARCH_full_FINAL_DATA_HEALTHY_SPORT_P.csv"
data = utils.load_data(DATA_PATH, FILE_NAME)
data = utils.rename_columns(data)
data = utils.clean_data(data)
print(data.columns)
# %%
columns_to_get = [
    'ID',
    'Sex',
    'Chronologic_Age',
    # 'Sport',
    # 'Sports',
    'Dominant_extremity',
    'Geographic_Factor',
    # 'QoL_EQ-5D-Y',
    'Pain_now',
    'Injury_History',
    'Injury_History_MoreThanOne (0=no,1=yes)',
    # 'Injury_History_Localization',
    # 'Injury_History_Localization_Upper_Lower_Torso',
    # 'Injury_History_Overuse_Acute',
    'Training_Volume_Weekly_MainSport',
    'Training_Volume_Weekly_ALLSports',
    # 'Age_started_main_sport',
    'Given_up_sport_for_main',
    'Main_sport_more_important',
    'Months_in_a_year>8',
    'Hours_per_week>Age',
    'Experience_main_sport',
    'Sports_Specialization',
    'Sports_Specialization_ordinal',
    # 'Weight (kg)',
    # 'Height (cm)',
    # 'BMI',
    # 'TK TłUSZCZ%',
    # 'Sitting_Height (cm)',
    # 'Age_PHV',
    # 'Maturity_Offset'
    ]

# Hand Held Dynamometry
HHD = [
    'PS_L_PEAK_FORCE',
    'PS_R_PEAK_FORCE',
    'CZ_L_PEAK_FORCE',
    'CZ_R_PEAK_FORCE',
    'DW_L_PEAK_FORCE',
    'DW_R_PEAK_FORCE',
    'BR_L_PEAK_FORCE',
    'BR_R_PEAK_FORCE'
    ]

# Y Balance Test
YBT = [
    'YBT_ANT_L_Normalized',
    'YBT_ANT_R_Normalized',
    'YBT_PM_L_Normalized',
    'YBT_PM_R_Normalized',
    'YBT_PL_L_Normalized',
    'YBT_PL_R_Normalized',
    'YBT_COMPOSITE_R',
    'YBT_COMPOSITE_L'
    ]

# Functional Movement Screen
FMS = ['FMS_TOTAL']

def process_data(d):
    d = utils.correct_ambiguity(d)
    d = utils.erase_rows(d)
    d = utils.correct_dtype(d)
    d = utils.recover_missing(d, 'BMI')
    d = utils.recover_missing(d, 'PHV')
    utils.print_columns(d)
    d = d[d['Pain_now'] == 0] # do not consider people with pain
    d = d.dropna()
    return d


data_pure = data[columns_to_get]
data_pure = process_data(data_pure)

data_HHD = data[columns_to_get + HHD]
data_HHD = process_data(data_HHD)

data_YBT = data[columns_to_get + YBT]
data_YBT = process_data(data_YBT)

data_FMS = data[columns_to_get + FMS]
data_FMS = process_data(data_FMS)

data_HHD_YBT = data[columns_to_get + HHD + YBT]
data_HHD_YBT = process_data(data_HHD_YBT)

data_HHD_FMS = data[columns_to_get + HHD + FMS]
data_HHD_FMS = process_data(data_HHD_FMS)

data_YBT_FMS = data[columns_to_get + YBT + FMS]
data_YBT_FMS = process_data(data_YBT_FMS)

data_HHD_YBT_FMS = data[columns_to_get + HHD + YBT + FMS]
data_HHD_YBT_FMS = process_data(data_HHD_YBT_FMS)

# Create new columns based on the dominant extremity
def assign_dominant_extremity(row, left_col, right_col):
    return row[left_col] if row['Dominant_extremity'] == 'Left' else row[right_col]


data_HHD['PS_DOMINANT_PEAK_FORCE'] = data_HHD.apply(assign_dominant_extremity, left_col='PS_L_PEAK_FORCE', right_col='PS_R_PEAK_FORCE', axis=1)
data_HHD['CZ_DOMINANT_PEAK_FORCE'] = data_HHD.apply(assign_dominant_extremity, left_col='CZ_L_PEAK_FORCE', right_col='CZ_R_PEAK_FORCE', axis=1)
data_HHD['DW_DOMINANT_PEAK_FORCE'] = data_HHD.apply(assign_dominant_extremity, left_col='DW_L_PEAK_FORCE', right_col='DW_R_PEAK_FORCE', axis=1)
data_HHD['BR_DOMINANT_PEAK_FORCE'] = data_HHD.apply(assign_dominant_extremity, left_col='BR_L_PEAK_FORCE', right_col='BR_R_PEAK_FORCE', axis=1)

data_HHD['PS_NONDOMINANT_PEAK_FORCE'] = data_HHD.apply(assign_dominant_extremity, right_col='PS_L_PEAK_FORCE', left_col='PS_R_PEAK_FORCE', axis=1)
data_HHD['CZ_NONDOMINANT_PEAK_FORCE'] = data_HHD.apply(assign_dominant_extremity, right_col='CZ_L_PEAK_FORCE', left_col='CZ_R_PEAK_FORCE', axis=1)
data_HHD['DW_NONDOMINANT_PEAK_FORCE'] = data_HHD.apply(assign_dominant_extremity, right_col='DW_L_PEAK_FORCE', left_col='DW_R_PEAK_FORCE', axis=1)
data_HHD['BR_NONDOMINANT_PEAK_FORCE'] = data_HHD.apply(assign_dominant_extremity, right_col='BR_L_PEAK_FORCE', left_col='BR_R_PEAK_FORCE', axis=1)


data_YBT['YBT_ANT_DOMINANT'] = data_YBT.apply(assign_dominant_extremity, left_col='YBT_ANT_L_Normalized', right_col='YBT_ANT_R_Normalized', axis=1)
data_YBT['YBT_PM_DOMINANT'] = data_YBT.apply(assign_dominant_extremity, left_col='YBT_PM_L_Normalized', right_col='YBT_PM_R_Normalized', axis=1)
data_YBT['YBT_PL_DOMINANT'] = data_YBT.apply(assign_dominant_extremity, left_col='YBT_PL_L_Normalized', right_col='YBT_PL_R_Normalized', axis=1)
data_YBT['YBT_COMPOSITE_DOMINANT'] = data_YBT.apply(assign_dominant_extremity, left_col='YBT_COMPOSITE_L', right_col='YBT_COMPOSITE_R', axis=1)

data_YBT['YBT_ANT_NONDOMINANT'] = data_YBT.apply(assign_dominant_extremity, right_col='YBT_ANT_L_Normalized', left_col='YBT_ANT_R_Normalized', axis=1)
data_YBT['YBT_PM_NONDOMINANT'] = data_YBT.apply(assign_dominant_extremity, right_col='YBT_PM_L_Normalized', left_col='YBT_PM_R_Normalized', axis=1)
data_YBT['YBT_PL_NONDOMINANT'] = data_YBT.apply(assign_dominant_extremity, right_col='YBT_PL_L_Normalized', left_col='YBT_PL_R_Normalized', axis=1)
data_YBT['YBT_COMPOSITE_NONDOMINANT'] = data_YBT.apply(assign_dominant_extremity, right_col='YBT_COMPOSITE_L', left_col='YBT_COMPOSITE_R', axis=1)
# %%

# Prepare data for linear regression models
data_HHD['Injury_History_MoreThanOne (0=no,1=yes)'] = data_HHD['Injury_History_MoreThanOne (0=no,1=yes)'].map({'Yes': 1, 'No': 0}).astype('category')
data_YBT['Injury_History_MoreThanOne (0=no,1=yes)'] = data_YBT['Injury_History_MoreThanOne (0=no,1=yes)'].map({'Yes': 1, 'No': 0}).astype('category')
data_FMS['Injury_History_MoreThanOne (0=no,1=yes)'] = data_FMS['Injury_History_MoreThanOne (0=no,1=yes)'].map({'Yes': 1, 'No': 0}).astype('category')
data_HHD['Injury_History'] = data_HHD['Injury_History'].map({'Yes': 1, 'No': 0}).astype('category')
data_YBT['Injury_History'] = data_YBT['Injury_History'].map({'Yes': 1, 'No': 0}).astype('category')
data_FMS['Injury_History'] = data_FMS['Injury_History'].map({'Yes': 1, 'No': 0}).astype('category')

data_HHD['Sex'] = data_HHD['Sex'].astype('category')
data_YBT['Sex'] = data_YBT['Sex'].astype('category')
data_FMS['Sex'] = data_FMS['Sex'].astype('category')

data_HHD['Sports_Specialization_ordinal'] = data_HHD['Sports_Specialization_ordinal'].astype('category')
data_YBT['Sports_Specialization_ordinal'] = data_YBT['Sports_Specialization_ordinal'].astype('category')
data_FMS['Sports_Specialization_ordinal'] = data_FMS['Sports_Specialization_ordinal'].astype('category')

data_HHD['Training_Volume_Weekly_ALLSports'] = data_HHD['Training_Volume_Weekly_ALLSports'].astype('float')
data_YBT['Training_Volume_Weekly_ALLSports'] = data_YBT['Training_Volume_Weekly_ALLSports'].astype('float')
data_FMS['Training_Volume_Weekly_ALLSports'] = data_FMS['Training_Volume_Weekly_ALLSports'].astype('float')



# %%
x = data_HHD[[
    'Sex',
    'Chronologic_Age',
    'Sports_Specialization_ordinal',
    'Training_Volume_Weekly_MainSport',
    'Training_Volume_Weekly_ALLSports',
    'Experience_main_sport',
    'Injury_History',
    'Injury_History_MoreThanOne (0=no,1=yes)'
    ]]
y = data_HHD[['PS_DOMINANT_PEAK_FORCE',
              'CZ_DOMINANT_PEAK_FORCE',
              'DW_DOMINANT_PEAK_FORCE',
              'BR_DOMINANT_PEAK_FORCE']]
# y = y[y.columns[0]]
# x = x[x.columns[1]]
# %%
import statsmodels.api as sm
from statsmodels.api import OLS

x = sm.add_constant(x)
model = OLS(y, x).fit()
print(model.summary())

# %%
