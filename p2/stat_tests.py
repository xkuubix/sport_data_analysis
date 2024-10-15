import sys
import os
sys.path.append(os.path.abspath('../')) 
import pandas as pd
import math
import random
import warnings
import logging
import utils
from sklearn.linear_model import LinearRegression, HuberRegressor
import numpy as np
# %%
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
warnings.simplefilter(action='ignore', category=FutureWarning)
DATA_PATH = "/media/dysk_a/jr_buler/Healthy_Sport"
FILE_NAME = "utf-8_RESEARCH_full_FINAL_DATA_HEALTHY_SPORT_P.csv"
data = utils.load_data(DATA_PATH, FILE_NAME)
data = utils.rename_columns(data)
data = utils.clean_data(data)
print(data.columns)
random.seed(42)
# %%
columns_to_get = [
    'ID',
    'Sex',
    'Chronologic_Age',
    'Sport',
    'Sports',
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

#%% tests
from scipy import stats

# check data normality and homogeneity of variance

YBT_KEYS = ['YBT_ANT_DOMINANT', 'YBT_PM_DOMINANT', 'YBT_PL_DOMINANT', 'YBT_COMPOSITE_DOMINANT',
            'YBT_ANT_NONDOMINANT', 'YBT_PM_NONDOMINANT', 'YBT_PL_NONDOMINANT', 'YBT_COMPOSITE_NONDOMINANT']

HHD_KEYS = ['PS_DOMINANT_PEAK_FORCE', 'CZ_DOMINANT_PEAK_FORCE', 'DW_DOMINANT_PEAK_FORCE', 'BR_DOMINANT_PEAK_FORCE',
            'PS_NONDOMINANT_PEAK_FORCE', 'CZ_NONDOMINANT_PEAK_FORCE', 'DW_NONDOMINANT_PEAK_FORCE', 'BR_NONDOMINANT_PEAK_FORCE']

FMS_KEYS = ['FMS_TOTAL']


# YBT DOMINANT VS NONDOMINANT BY SPORTS SPECIALIZATION
for key in YBT_KEYS:
    for spec in data_YBT['Sports_Specialization'].unique():
        stat, pvalue = stats.shapiro(data_YBT[key][data_YBT['Sports_Specialization'] == spec])
        n = len(data_YBT[key][data_YBT['Sports_Specialization'] == spec])
        n = ' (' + str(n) + ')'
        print(f'{key:25s} {n:5s} {spec:10s} Shapiro-Wilk test: {stat:.3f}, {pvalue:.6f}')
    stat, pvalue = stats.levene(data_YBT[key][data_YBT['Sports_Specialization'] == 'low'],
                                data_YBT[key][data_YBT['Sports_Specialization'] == 'moderate'],
                                data_YBT[key][data_YBT['Sports_Specialization'] == 'high'])
    print(f'Levene test: {stat:.3f}, {pvalue:.6f}')
    print('\n')


# HDD DOMINANT VS NONDOMINANT BY SPORTS SPECIALIZATION
print('\n')
for key in HHD_KEYS:
    for spec in data_HHD['Sports_Specialization'].unique():
        stat, pvalue = stats.shapiro(data_HHD[key][data_HHD['Sports_Specialization'] == spec])
        n = len(data_HHD[key][data_HHD['Sports_Specialization'] == spec])
        n = ' (' + str(n) + ')'
        print(f'{key:25s} {n:5s} {spec:10s} Shapiro-Wilk test: {stat:.3f}, {pvalue:.6f}')
    stat, pvalue = stats.levene(data_HHD[key][data_HHD['Sports_Specialization'] == 'low'],
                                data_HHD[key][data_HHD['Sports_Specialization'] == 'moderate'],
                                data_HHD[key][data_HHD['Sports_Specialization'] == 'high'])
    print(f'Levene test: {stat:.3f}, {pvalue:.6f}')
    print('\n')

print('\n')

# FMS BY SPORTS SPECIALIZATION
for key in FMS_KEYS:
    for spec in data_FMS['Sports_Specialization'].unique():
        stat, pvalue = stats.shapiro(data_FMS[key][data_FMS['Sports_Specialization'] == spec])
        n = len(data_FMS[key][data_FMS['Sports_Specialization'] == spec])
        n = ' (' + str(n) + ')'
        print(f'{key:25s} {n:5s} {spec:10s} Shapiro-Wilk test: {stat:.3f}, {pvalue:.6f}')
    stat, pvalue = stats.levene(data_FMS[key][data_FMS['Sports_Specialization'] == 'low'],
                                data_FMS[key][data_FMS['Sports_Specialization'] == 'moderate'],
                                data_FMS[key][data_FMS['Sports_Specialization'] == 'high'])
    print(f'Levene test: {stat:.3f}, {pvalue:.6f}')
    print('\n')

# %%
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scikit_posthocs import posthoc_dunn

def multiple_means_and_post_hocs(data, keys, group_col='Sports_Specialization'):
    for key in keys:
        print(f"\nTesting for: {key} between groups in {group_col}")

        # Descriptive statistics
        for group in data[group_col].unique():
            print(f"Group {group}:")
            print(f"{data[key][data[group_col] == group].mean():.2f}", end=' (')
            print(f"{data[key][data[group_col] == group].std():.2f})")

        # Get data for each Sport Specialization group
        group_data = [data[key][data[group_col] == group] 
                      for group in data[group_col].unique()]
    
        # Step 1: Test for Normality (Shapiro-Wilk)
        normality_pvalues = [stats.shapiro(group)[1] for group in group_data]
        normality = all(p > 0.05 for p in normality_pvalues)  # True if all groups are normally distributed
    
        # Step 2: Test for Homogeneity of Variances (Levene's Test)
        stat, pvalue_levene = stats.levene(*group_data)
        homogeneity = pvalue_levene > 0.05  # True if variances are equal

        # Step 3: Choose ANOVA or Kruskal-Wallis
        if normality and homogeneity:
        # Perform ANOVA
            stat, pvalue_anova = stats.f_oneway(*group_data)
            print(f"ANOVA result for {key}: p-value = {pvalue_anova}")
            if pvalue_anova < 0.05:
                print(f"Significant differences found for {key}, performing Tukey's HSD post-hoc test.")
            # Perform Tukey HSD
                posthoc = pairwise_tukeyhsd(data[key], data[group_col], alpha=0.05)
                print(posthoc)
        else:
        # Perform Kruskal-Wallis Test (non-parametric)
            stat, pvalue_kruskal = stats.kruskal(*group_data)
            print(f"Kruskal-Wallis result for {key}: p-value = {pvalue_kruskal}")
            if pvalue_kruskal < 0.05:
                print(f"Significant differences found for {key}, performing Dunn's post-hoc test.")
            # Perform Dunns post-hoc
                posthoc = posthoc_dunn(data, val_col=key, group_col=group_col, p_adjust='bonferroni')
                print(posthoc)



multiple_means_and_post_hocs(data=data_YBT, keys=YBT_KEYS, group_col='Sports_Specialization')
multiple_means_and_post_hocs(data=data_HHD, keys=HHD_KEYS, group_col='Sports_Specialization')
multiple_means_and_post_hocs(data=data_FMS, keys=FMS_KEYS, group_col='Sports_Specialization')

# %%

print(data_YBT['Sports_Specialization'].value_counts())
print(data_HHD['Sports_Specialization'].value_counts())
print(data_FMS['Sports_Specialization'].value_counts())


# %%
# pearson

def two_means_correlation(data, column1, column2):
    print(f"\nCorrelation analysis: {column1} and {column2}")
    pearson_corr, p_value = stats.pearsonr(data[column1], data[column2])
    print(f"Pearson correlation: {pearson_corr:.3f}, p-value: {p_value:.6f}")
    spearman_corr, p_value = stats.spearmanr(data[column1], data[column2])
    print(f"Spearman correlation: {spearman_corr:.3f}, p-value: {p_value:.6f}")

#%%
# YBT
for key in YBT_KEYS:
    two_means_correlation(data_YBT, key, 'Training_Volume_Weekly_MainSport')
for key in YBT_KEYS:
    two_means_correlation(data_YBT, key, 'Training_Volume_Weekly_ALLSports')

for key in HHD_KEYS:
    two_means_correlation(data_HHD, key, 'Training_Volume_Weekly_MainSport')
for key in HHD_KEYS:
    two_means_correlation(data_HHD, key, 'Training_Volume_Weekly_ALLSports')

for key in FMS_KEYS:
    two_means_correlation(data_FMS, key, 'Training_Volume_Weekly_MainSport')
for key in FMS_KEYS:
    two_means_correlation(data_FMS, key, 'Training_Volume_Weekly_ALLSports')

# %%

def perform_ttest_or_mannwhitney(data, group_col, value_col):
    groups = data[group_col].unique()
    if len(groups) != 2:
        raise ValueError("The group column must have exactly two unique values for this test.")
    
    group1 = data[data[group_col] == groups[0]][value_col]
    group2 = data[data[group_col] == groups[1]][value_col]
    
    for group in groups:
        print(f"Group {group}:")
        print(f"{data[key][data[group_col] == group].mean():.2f}", end=' (')
        print(f"{data[key][data[group_col] == group].std():.2f})")
    # Test for normality
    stat1, pvalue1 = stats.shapiro(group1)
    stat2, pvalue2 = stats.shapiro(group2)
    normality = pvalue1 > 0.05 and pvalue2 > 0.05
    
    # Test for homogeneity of variances
    stat, pvalue_levene = stats.levene(group1, group2)
    homogeneity = pvalue_levene > 0.05
    
    if normality and homogeneity:
        # Perform Student's t-test
        stat, pvalue = stats.ttest_ind(group1, group2)
        test_name = "Student's t-test"
    else:
        # Perform Mann-Whitney U test
        stat, pvalue = stats.mannwhitneyu(group1, group2)
        test_name = "Mann-Whitney U test"
    
    print(f"{test_name} result for {value_col} between {groups[0]} and {groups[1]}: stat = {stat:.3f}, p-value = {pvalue:.6f}")
    if pvalue < 0.05:
        print('\n')
    return stat, pvalue
# %%
for key in YBT_KEYS:
    perform_ttest_or_mannwhitney(data_YBT, 'Hours_per_week>Age', key)
# %%
for key in HHD_KEYS:
    perform_ttest_or_mannwhitney(data_HHD, 'Hours_per_week>Age', key)

# %%
for key in FMS_KEYS:
    perform_ttest_or_mannwhitney(data_FMS, 'Hours_per_week>Age', key)

# %%
print(data_YBT['Hours_per_week>Age'].value_counts())
print(data_HHD['Hours_per_week>Age'].value_counts())
print(data_FMS['Hours_per_week>Age'].value_counts())

# %%
for key in YBT_KEYS:
    perform_ttest_or_mannwhitney(data_YBT, 'Injury_History', key)

# %%
for key in HHD_KEYS:
    perform_ttest_or_mannwhitney(data_HHD, 'Injury_History', key)

# %%
for key in FMS_KEYS:
    perform_ttest_or_mannwhitney(data_FMS, 'Injury_History', key)

# %%
print(data_YBT['Injury_History'].value_counts())
print(data_HHD['Injury_History'].value_counts())
print(data_FMS['Injury_History'].value_counts())

# %%
for key in YBT_KEYS:
    perform_ttest_or_mannwhitney(data_YBT, 'Injury_History_MoreThanOne (0=no,1=yes)', key)

# %%
for key in HHD_KEYS:
    perform_ttest_or_mannwhitney(data_HHD, 'Injury_History_MoreThanOne (0=no,1=yes)', key)

# %%
for key in FMS_KEYS:
    perform_ttest_or_mannwhitney(data_FMS, 'Injury_History_MoreThanOne (0=no,1=yes)', key)

# %%
print(data_YBT['Injury_History_MoreThanOne (0=no,1=yes)'].value_counts())
print(data_HHD['Injury_History_MoreThanOne (0=no,1=yes)'].value_counts())
print(data_FMS['Injury_History_MoreThanOne (0=no,1=yes)'].value_counts())


# %%
for key in YBT_KEYS:
    two_means_correlation(data_YBT, key, 'Chronologic_Age')

for key in HHD_KEYS:
    two_means_correlation(data_HHD, key, 'Chronologic_Age')

for key in FMS_KEYS:
    two_means_correlation(data_FMS, key, 'Chronologic_Age')


# %% SPORTS SPECIALIZATION CRITERIA
# # 4 sport specialization criteria:
# 'Given_up_sport_for_main', 'Main_sport_more_important',
# 'Months_in_a_year>8', 'Hours_per_week>Age',
criteria = ['Given_up_sport_for_main', 'Main_sport_more_important', 'Months_in_a_year>8', 'Hours_per_week>Age']
num = 3
print(criteria[num])
for key in YBT_KEYS:
    perform_ttest_or_mannwhitney(data_YBT, criteria[num], key)
for key in HHD_KEYS:
    perform_ttest_or_mannwhitney(data_HHD, criteria[num], key)
for key in FMS_KEYS:
    perform_ttest_or_mannwhitney(data_FMS, criteria[num], key)

# %%