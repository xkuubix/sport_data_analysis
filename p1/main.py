#%%
import os
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.tri as tri
import seaborn as sns
from scipy import stats
import warnings
import scipy.stats as stats
import logging
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.formula.api import logit
# %%
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_theme()

DATA_PATH = "/media/dysk_a/jr_buler/Healthy_Sport"
FILE_NAME = "utf-8_RESEARCH_full_FINAL_DATA_HEALTHY_SPORT_P.csv"
#%%
default_missing = pd._libs.parsers.STR_NA_VALUES
default_missing = default_missing.remove('')
data = pd.read_csv(os.path.join(DATA_PATH, FILE_NAME),
                   delimiter=';', encoding='utf-8',
                   na_filter=True, na_values=default_missing)

data.rename(columns={"Sports (1 = individual, 2 = team sports)": "Sports"}, inplace=True)
data["Sports"] = data["Sports"].map({1: "individual", 2: "team"})

data.rename(columns={"Sports_Specialization ":"Sports_Specialization"}, inplace=True)
data.rename(columns={"Athletes who participated in their primary sport for more hours per week than their age (Yes/No)": "Hours_per_week>Age"}, inplace=True)
data.rename(columns={"Injury_History (yes = 1, no = 0)": "Injury_History"}, inplace=True)
data["Injury_History"] = data["Injury_History"].map({1: "Yes", 0: "No"})

data.rename(columns={"Have you trained in a main sport for more than 8 months in one year?": "Months_in_a_year>8"}, inplace=True)
data["Months_in_a_year>8"] = data["Months_in_a_year>8"].map(lambda x: "Yes" if x.lower().startswith('tak') else "No", na_action="ignore")

data.rename(columns={'At what age did you start your main sport? (years)': 'Age_started_main_sport'}, inplace=True)
data.rename(columns={'Training_Volume_Weekly_MainSport (hrs)': 'Training_Volume_Weekly_MainSport'}, inplace=True)
data.rename(columns={'Training_Volume_Weekly_ALLSports (hrs)': 'Training_Volume_Weekly_ALLSports'}, inplace=True)
data.rename(columns={'Sex_(M=1, F=2)': 'Sex'}, inplace=True)
# print(data.columns)

data['Have you given up a sport for your main sport?'] = data['Have you given up a sport for your main sport?'].map(lambda x: "Yes" if x.lower().startswith('tak') else "No", na_action="ignore")
data['Is your main sport significantly more important than other sports?'] = data['Is your main sport significantly more important than other sports?'].map(lambda x: "Yes" if x.lower().startswith('tak') else "No", na_action="ignore")

data.rename(columns={'Experience_main_sport (years)': 'Experience_main_sport'}, inplace=True)

data["Sport"] = data["Sport"].str.lower()
data["Sport"] = data["Sport"].str.replace("teakwondo", "taekwondo")
#%%
subset_data = data.iloc[:, 0:25]
subset = subset_data.drop(columns=["QoL - EQ-5D-Y",
                                   "Dominant_extremity",])
                                #    "TK TłUSZCZ%"])

# if we have empty BMI, we can calculate it with formula: weight / (height^2)
# if we have empty height, we can calculate it with formula: sqrt(weight / BMI) * 100
# if we have empty weight, we can calculate it with formula: BMI * (height^2)
# subset.loc[subset['BMI'].isna(), 'BMI'] = subset['Weight (kg)'] / ((subset['Height (cm)'] / 100) ** 2)
# subset.loc[subset['Height (cm)'].isna(), 'Height (cm)'] = np.sqrt(subset.loc[subset['Height (cm)'].isna(), 'Weight (kg)'] / subset.loc[subset['Height (cm)'].isna(), 'BMI']) * 100
# subset.loc[subset['Weight (kg)'].isna(), 'Weight (kg)'] = subset.loc[subset['Weight (kg)'].isna(), 'BMI'] * ((subset.loc[subset['Weight (kg)'].isna(), 'Height (cm)'] / 100) ** 2)

# age phv calculation
# subset["Leg_length (cm)"] = subset["Height (cm)"] - subset["Sitting_Height (cm)"]
# subset["Leg_length_sitting_height_interaction"] = subset["Leg_length (cm)"] * subset["Sitting_Height (cm)"]
# subset["Age_leg_Length_interaction"] = subset["Leg_length (cm)"] * subset["Chronologic_Age"]
# subset["Age_sitting_height_interaction"] = subset["Chronologic_Age"] * subset["Sitting_Height (cm)"]
# subset["Age_weight_interaction"] = subset["Chronologic_Age"] * subset["Weight (kg)"]
# subset['Weight_height_ratio'] = subset['Weight (kg)'] / subset['Height (cm)'] * 100
# subset["Sitting_Standing_Height_Ratio"] = subset["Sitting_Height (cm)"] / subset["Height (cm)"]
# subset["Maturity_offset_calculated"] = np.where(subset["Sex"] == 2,
#                                                (-9.376 + (0.0001882 * subset["Leg_length_sitting_height_interaction"]) + (0.0022 * subset["Age_leg_Length_interaction"]) + (0.005841 * subset["Age_sitting_height_interaction"]) + (-0.002658 * subset["Age_weight_interaction"]) + (0.07693 * subset["Weight_height_ratio"])),
#                                                (-9.3236 + (0.0002708 * subset["Leg_length_sitting_height_interaction"]) + (-0.001663 * subset["Age_leg_Length_interaction"]) + (0.007216 * subset["Age_sitting_height_interaction"]) + (0.02292 * subset["Weight_height_ratio"])))
# subset = subset[subset['Maturity_Offset (years)'] != "A"] # probably a mistake (NA)
# subset["Maturity_Offset (years)"] = subset["Maturity_Offset (years)"].astype(float)
# # subset["Maturity_Offset_Difference"] = subset["Maturity_offset_calculated"] - subset["Maturity_Offset (years)"]
# subset["Age_PHV_calculated"] = subset["Chronologic_Age"] - subset["Maturity_offset_calculated"]

# subset = subset.drop(columns=["Age_PHV",
#                               "Maturity_Offset (years)"])
# drop rows with ambiguous injury information
subset = subset[((subset["Injury_History"] == "No") & (subset["Injury_History_Localization_Upper_Lower_Torso"] == "") & (subset["Injury_History_Overuse_Acute"] == "")) | ((subset["Injury_History"] == "Yes") & (subset["Injury_History_Overuse_Acute"] != "") & (subset["Injury_History_Overuse_Acute"] != ""))]

subset = subset.dropna()

# %%
subset = subset[subset.Injury_History_Localization_Upper_Lower_Torso != "Head"]
subset = subset[subset.Injury_History_Localization_Upper_Lower_Torso != "Neck"]

# %%
# set empty values in Training_Volume_Weekly_ALLSports to Training_Volume_Weekly_MainSport
subset.loc[subset["Training_Volume_Weekly_ALLSports"] == '', "Training_Volume_Weekly_ALLSports"] = subset.loc[subset["Training_Volume_Weekly_ALLSports"] == '', "Training_Volume_Weekly_MainSport"]

subset["Training_Volume_Weekly_ALLSports"] = subset["Training_Volume_Weekly_ALLSports"].astype(float)
subset["Training_Volume_Weekly_MainSport"] = subset["Training_Volume_Weekly_MainSport"].astype(float)
subset["Age_started_main_sport"] = subset["Age_started_main_sport"].astype(float)
subset["Chronologic_Age"] = subset["Chronologic_Age"].astype(float)
subset["Experience_main_sport"] = subset["Experience_main_sport"].astype(float)

ind = subset[subset["Sports"] == "individual"]
team = subset[subset["Sports"] == "team"]
ordinal_mapping = {'low': 1, 'moderate': 2, 'high': 3}
subset['Spec_ordinal'] = subset['Sports_Specialization'].map(ordinal_mapping)

# subset = subset[(subset["Injury_History"] == "No") & (subset["Injury_History_Localization"] == "") | (subset["Injury_History"] == "Yes") & (subset["Injury_History_Localization"] != "")]
# subset["Injury_History"] = subset["Injury_History_Localization"] != ""
# subset["Injury_History"] = subset["Injury_History"].map({True: "Yes", False: "No"})

# %%

def indepentend_test(group1, group2, column_name, test_type, test_side="two-sided"):
    """
    Perform a statistical test to compare two independent groups.

    Parameters:
    - group1 (DataFrame): The first group for comparison.
    - group2 (DataFrame): The second group for comparison.
    - column_name (str): The column name to be used for comparison.
    - test_type (str): The type of test to be performed. Valid options are "2-means" and "proportions".
    - test_side (str, optional): The side of the test. Valid options are "two-sided", "greater", or "less". Default is "two-sided".

    Returns:
    - res (tuple): The result of the statistical test. The format of the result depends on the test type:
        - For "2-means" test: (test statistic, p-value)
        - For "proportions" test: (chi-square statistic, p-value, degrees of freedom, expected frequencies)

    """
    if test_type == "2-means":
        logger.info(f'Testing independent groups for means ({test_side=}) ...')
        # Check homogeneity of variances assumption (Levene's test)
        logger.info(f'Performing Levene\'s test for homogeneity of variances ...')
        if stats.levene(group1[column_name], group2[column_name]).pvalue < 0.05:
            equal_var = False
        else:
            equal_var = True
        logger.info(f'Equal variances assumption: {equal_var}, p-value = {stats.levene(group1[column_name], group2[column_name])[1]}')
        # Check sample sizes
        logger.info(f'Sample sizes: {group1[column_name].count()}, {group2[column_name].count()}')

        # Check normality assumption
        if stats.shapiro(group1[column_name]).pvalue < 0.05 or stats.shapiro(group2[column_name]).pvalue < 0.05:
            logger.info(f'p-value group1: {stats.shapiro(group1[column_name]).pvalue}, p-value group2: {stats.shapiro(group2[column_name]).pvalue}')
            logger.info('Normality assumption is not met, performing non-parametric test (Mann-Whitney U test) ...')
            # Mann-Whitney U test
            res = stats.mannwhitneyu(group1[column_name], group2[column_name], alternative=test_side)
        else:
            # Welch's t-test (adjusted for unequal variances and sample sizes) or Student's t-test
            type_t_test = "adjusted Welch's" if not equal_var else "Student's"
            logger.info(f'p-value group1: {stats.shapiro(group1[column_name])[1]}, p-value group2: {stats.shapiro(group2[column_name])[1]}')
            logger.info(f'Normality assumption met, performing ({type_t_test}) t-test ...')
            res = stats.ttest_ind(group1[column_name], group2[column_name], equal_var=equal_var, alternative=test_side)

    elif test_type == "proportions":
        # Chi-square test of independence
        group1_counts = group1[column_name].value_counts().to_dict()
        group2_counts = group2[column_name].value_counts().to_dict()
        fisher = 0
        for k in group1_counts.keys():
            if k not in group2_counts:
                group2_counts[k] = 0
                fisher = 1
        for k in group2_counts.keys():
            if k not in group1_counts:
                group1_counts[k] = 0
                fisher = 1
        group1_counts = dict(sorted(group1_counts.items()))
        group2_counts = dict(sorted(group2_counts.items()))
        group1_counts = [v for _, v in group1_counts.items()]
        group2_counts = [v for _, v in group2_counts.items()]
        print(f'{group1_counts=}')
        print(f'{group2_counts=}')    
        for k in group1_counts:
            if k < 5:
                fisher = 1
        for k in group2_counts:
            if k < 5:
                fisher = 1
        if fisher:
            res = stats.fisher_exact([group1_counts, group2_counts])
        else:
            res = stats.chi2_contingency([group1_counts, group2_counts])
    
    return res

# %%
keys = ['Sex', 'Chronologic_Age', 'Sports',
    'Geographic_Factor', 'Pain_now (0=NO, 1=YES)', 'Injury_History',
    'Experience_main_sport',
    'Injury_History_Localization_Upper_Lower_Torso',
    'Injury_History_Overuse_Acute',
    'Injury_History_MoreThanOne (0=no,1=yes)',
    'Training_Volume_Weekly_MainSport', 'Training_Volume_Weekly_ALLSports',
    'Age_started_main_sport', 'Hours_per_week>Age',
    'Have you given up a sport for your main sport?',
    'Is your main sport significantly more important than other sports?',
    'Months_in_a_year>8', 'Sports_Specialization',
    'Sports_Specialization (points) (0-1 low, 2 moderate, 3 high)',
    'Weight (kg)', 'Height (cm)', 'BMI', 'Sitting_Height (cm)',
    'Leg_length (cm)', 'Leg_length_sitting_height_interaction',
    'Age_leg_Length_interaction', 'Age_sitting_height_interaction',
    'Age_weight_interaction', 'Weight_height_ratio',
    'Sitting_Standing_Height_Ratio', 'Maturity_offset_calculated',
    'Age_PHV_calculated', 'Spec_ordinal']



pairs = [('Sex', 'proportions'),
         ('Chronologic_Age', '2-means'),
         ('Age_started_main_sport', '2-means'),
         ('Experience_main_sport', '2-means'),
         ('Training_Volume_Weekly_ALLSports', '2-means'),
         ('Training_Volume_Weekly_MainSport', '2-means'),
         ('Hours_per_week>Age', 'proportions'),
         ('Have you given up a sport for your main sport?', 'proportions'),
         ('Is your main sport significantly more important than other sports?', 'proportions'),
         ('Months_in_a_year>8', 'proportions'),
         ('Sports_Specialization (points) (0-1 low, 2 moderate, 3 high)', '2-means'),
         ('Injury_History', 'proportions'),
         ('Injury_History_Overuse_Acute', 'proportions'),
         ('Injury_History_MoreThanOne (0=no,1=yes)', 'proportions'),
         ('Injury_History_Localization_Upper_Lower_Torso', 'proportions'),
         ('Geographic_Factor', 'proportions')]

for key, test_type in pairs:
    print(f'\n{key=}')
    res = indepentend_test(ind, team, column_name=key,
                           test_type=test_type)
    print(f'{res=}')
# %%
group = "Sports"
for key in keys:
    if key in subset.columns:
        print('\n')
        print(f'==================={key}===================')
        if len(subset[key].unique()) <= 4:
            print('\n')
            print('\nTotal')
            counts = subset[key].value_counts()
            percentages = round(subset[key].value_counts(normalize=True) * 100,1)
            print(pd.DataFrame({
                'Counts': counts,
                'Percentages': percentages
            }).to_string())
            counts = subset.groupby(group)[key].value_counts()
            percentages = round(subset.groupby(group)[key].value_counts(normalize=True) * 100,1)
            print(pd.DataFrame({
                'Counts': counts,
                'Percentages': percentages
            }).to_string())
        else:
            print('\n')
            print('\n')
            mean = subset.groupby(group)[key].mean()
            std = subset.groupby(group)[key].std()
            for g, m, s in zip(mean.index, mean, std):
                    print(f"Mean +- std for {g}: {m:.2f}±{s:.2f}")
            mean = subset[key].mean()
            std = subset[key].std()
            print(f"Mean +- std for total: {mean:.2f}±{std:.2f}")
    else:
        print(f'{key} column not found in the dataframe.')




# %%
# top_10_sports = subset["Sport"].value_counts().nlargest(10).index
# data_top_10 = subset[subset["Sport"].isin(top_10_sports)]
# subset = data_top_10


d = subset[subset['Months_in_a_year>8'] == "Yes"]
top_10_sports = d["Sport"].value_counts().nlargest(10).index
data_top_10 = d[subset["Sport"].isin(top_10_sports)]
subset = data_top_10


for key in keys:
    if key in subset.columns:
        print('\n')
        print(f'==================={key}===================')
        if len(subset[key].unique()) <= 4:
            print('\n')
            counts = subset.groupby('Sport')[key].value_counts()
            percentages = round(subset.groupby('Sport')[key].value_counts(normalize=True) * 100,1)
            print(pd.DataFrame({
                'Counts': counts,
                'Percentages': percentages
            }).to_string())
        else:
            print('\n')
            print('\n')
            mean = subset.groupby('Sport')[key].mean()
            std = subset.groupby('Sport')[key].std()
            for g, m, s in zip(mean.index, mean, std):
                    print(f"Mean +- std for {g}: {m:.2f}±{s:.2f}")
            mean = subset[key].mean()
            std = subset[key].std()
            print(f"Mean +- std for total: {mean:.2f}±{std:.2f}")
    else:
        print(f'{key} column not found in the dataframe.')


# mean = subset.groupby("Sports")["Training_Volume_Weekly_ALLSports"].mean()
# std = subset.groupby("Sports")["Training_Volume_Weekly_ALLSports"].std()

# for sport, m, s in zip(mean.index, mean, std):
#     print(f"{sport}: {m:.2f}±{s:.2f}")



# %%
# H1 – Zawodnicy z indywidualnych sportów („Sports (1 = individual, 2 = team sports)” )
# będą charakteryzować się wyższą (high) specjalizacją niż zawodnicy sportów zespołowych

# Chronologic_Age +  + Training_Volume_Weekly_ALLSports'

res = indepentend_test(ind, team, column_name="Sports_Specialization (points) (0-1 low, 2 moderate, 3 high)",
                       test_type="2-means")
print(f'\n{res=}')


# %%

ind['Spec_ordinal'] = ind['Sports_Specialization'].map(ordinal_mapping)
team['Spec_ordinal'] = team['Sports_Specialization'].map(ordinal_mapping)

k = "Sports_Specialization (points) (0-1 low, 2 moderate, 3 high)"
res = stats.mannwhitneyu(ind[k], team[k],
                         alternative='two-sided')
# %%
# SPORT INDYWIUDALNY VS SPORT ZESPOŁOWY

def describe_data(data):
    desc = stats.describe(data)
    print(f'Mean = {desc.mean}')
    print(f'STD = {math.sqrt(desc.variance)}')
    print(f'Var = {desc.variance}')
    print(f'Skew = {desc.skewness}')
    print(f'Kurt = {desc.kurtosis}')

keys = ["Age_started_main_sport",
        # "Height (cm)", "Weight (kg)", "BMI",
        "Chronologic_Age",
        "Experience_main_sport", "Training_Volume_Weekly_MainSport",
        "Training_Volume_Weekly_ALLSports"]

# keys = ["Experience_main_sport"]
for k in keys:
    print(f'\n{k}')
    print("\nAll")
    describe_data(subset[k].dropna())
    print("\nIndividual")
    describe_data(ind[k].dropna())
    print("\nTeam")
    describe_data(team[k].dropna())

# %%
# res = stats.levene(ind["Chronologic_Age"], team["Chronologic_Age"])
# print(res)
# res = stats.levene(ind["Age_started_main_sport"], team["Age_started_main_sport"])
# print(res)
# res = stats.levene(ind["Experience_main_sport"], team["Experience_main_sport"])
# print(res)
# res = stats.levene(ind["Training_Volume_Weekly_ALLSports"], team["Training_Volume_Weekly_ALLSports"])
# print(res)
# res = stats.levene(ind["Training_Volume_Weekly_MainSport"], team["Training_Volume_Weekly_MainSport"])
# print(res)


res = indepentend_test(ind, team, column_name="Chronologic_Age",
                       test_type="2-means", test_side='two-sided')
print(f'\n{res.statistic=}\n{res.pvalue=}')



#%%
from statsmodels.multivariate.manova import MANOVA

# Sample data (replace with your actual data)

# df['Group'] = df['Group'].astype('category')
# MANOVA
# manova = MANOVA.from_formula('Sports ~ Chronologic_Age + Age_started_main_sport + Training_Volume_Weekly_ALLSports + Training_Volume_Weekly_MainSport', data=subset)
# print(manova.mv_test())

subset["Sports_numeric"] = subset["Sports"].map({"individual": 0, "team": 1})
model_adjusted = smf.logit('Sports_numeric ~ Chronologic_Age + Training_Volume_Weekly_MainSport + Training_Volume_Weekly_ALLSports', data=subset).fit()
print(model_adjusted.summary())
# %%
df = subset
df['low'] = df['Sports_Specialization'].map(lambda x: 1 if x == 'low' else 0)
df['moderate'] = df['Sports_Specialization'].map(lambda x: 1 if x == 'moderate' else 0)
df['high'] = df['Sports_Specialization'].map(lambda x: 1 if x == 'high' else 0)


# Logistic regression for low spec
model_low = smf.logit('Sports_numeric ~  low + Chronologic_Age + Sex + Training_Volume_Weekly_MainSport', data=df).fit()
# model_low = smf.logit('Sports_numeric ~  low', data=df).fit()
print(model_low.summary())

# Logistic regression for moderate spec
model_moderate= smf.logit('Sports_numeric ~  moderate + Chronologic_Age + Sex + Training_Volume_Weekly_MainSport', data=df).fit()
# model_moderate = smf.logit('Sports_numeric ~  moderate', data=df).fit()
print(model_moderate.summary())

# Logistic regression for high spec
model_high = smf.logit('Sports_numeric ~  high + Chronologic_Age + Sex + Training_Volume_Weekly_MainSport', data=df).fit()
# model_high = smf.logit('Sports_numeric ~  high', data=df).fit()
print(model_high.summary())


#%%
ind_injured_counts = ind['Injury_History'].value_counts().to_dict() 
ind_injured_counts = dict(sorted(ind_injured_counts.items()))
print(ind_injured_counts)
ind_injured_counts = [v for _, v in ind_injured_counts.items()]
team_injured_counts = team['Injury_History'].value_counts().to_dict()
team_injured_counts = dict(sorted(team_injured_counts.items()))
print(team_injured_counts)
team_injured_counts = [v for _, v in team_injured_counts.items()]
res = stats.chi2_contingency([ind_injured_counts, team_injured_counts])
print(res)
#%%
df = subset[subset['Injury_History'] == 'Yes']
df_ind = ind[ind['Injury_History'] == 'Yes']
print(df_ind['Injury_History_Localization_Upper_Lower_Torso'].value_counts())
print(round(100 * df_ind['Injury_History_Localization_Upper_Lower_Torso'].value_counts() / df_ind['Injury_History_Localization_Upper_Lower_Torso'].count(), 2))
print('---')
df_team = team[team['Injury_History'] == 'Yes']
print(df_team['Injury_History_Localization_Upper_Lower_Torso'].value_counts())
print(round(100 * df_team['Injury_History_Localization_Upper_Lower_Torso'].value_counts() / df_team['Injury_History_Localization_Upper_Lower_Torso'].count(), 2))
print('---')
print(df['Injury_History_Localization_Upper_Lower_Torso'].value_counts())
print(round(100 * df['Injury_History_Localization_Upper_Lower_Torso'].value_counts() / df['Injury_History_Localization_Upper_Lower_Torso'].count(), 2))

# %%
subset['lower'] = subset['Injury_History_Localization_Upper_Lower_Torso'].map(lambda x: 1 if x == 'Lower' else 0)
subset['upper'] = subset['Injury_History_Localization_Upper_Lower_Torso'].map(lambda x: 1 if x == 'Upper' else 0)
subset['torso'] = subset['Injury_History_Localization_Upper_Lower_Torso'].map(lambda x: 1 if x == 'Torso' else 0)
df = subset[subset['Injury_History'] == 'Yes']

# Logistic regression for lower injury
model_lower = smf.logit('Sports_numeric ~  lower + Chronologic_Age + Sex + Training_Volume_Weekly_MainSport', data=df).fit()
# model_lower = smf.logit('Sports_numeric ~  lower', data=df).fit()
print(model_lower.summary())

# Logistic regression for upper injury
model_upper = smf.logit('Sports_numeric ~  upper + Chronologic_Age + Sex + Training_Volume_Weekly_MainSport', data=df).fit()
# model_upper = smf.logit('Sports_numeric ~  upper', data=df).fit()
print(model_upper.summary())

# Logistic regression for torso injury
model_torso = smf.logit('Sports_numeric ~  torso + Chronologic_Age + Sex + Training_Volume_Weekly_MainSport', data=df).fit()
# model_torso = smf.logit('Sports_numeric ~  torso', data=df).fit()
print(model_torso.summary())

# %%
# -------------------
df = pd.DataFrame(subset)

subset["Sports_numeric"] = subset["Sports"].map({"individual": 0, "team": 1})
covariates = ['Spec_ordinal', 'Chronologic_Age', 'Training_Volume_Weekly_MainSport']
outcomes = ['Sports_numeric']
# Create binary columns for each injury type


# Define the independent variables (including the constant term)
X = df[covariates]
X = sm.add_constant(X)  # Adds a constant term to the predictor

# Logistic regression for general injury
y_injury = df['Sports']
model_injury = sm.Logit(y_injury, X)
result_injury = model_injury.fit()
print("General Injury Logistic Regression Results:")
print(result_injury.summary())

# Logistic regression for overuse injury
y_overuse = df['Overuse_Injury']
model_overuse = sm.Logit(y_overuse, X)
result_overuse = model_overuse.fit()
print("\nOveruse Injury Logistic Regression Results:")
print(result_overuse.summary())

# Logistic regression for acute injury
y_acute = df['Acute_Injury']
model_acute = sm.Logit(y_acute, X)
result_acute = model_acute.fit()
print("\nAcute Injury Logistic Regression Results:")
print(result_acute.summary())



# %%

keys = {1: "Sports_Specialization", 2: "Sex", 3: 'Injury_History_Overuse_Acute', 4: 'Injury_History_MoreThanOne (0=no,1=yes)'}
k = keys[3]
d = subset
print("\nAll\n")
counts = d[k].value_counts()
del counts['']
counts_sum = counts.iloc[0] + counts.iloc[1]
print(counts)
print(round(100*(counts/counts_sum),2))
print('\nindividual\n')
counts = ind[k].value_counts()
del counts['']
ind_counts = counts
counts_sum = counts.iloc[0] + counts.iloc[1]
print(counts)
print(round(100*(counts/counts_sum),2))
print('\nteam\n')
counts = team[k].value_counts()
del counts['']
team_counts = counts
counts_sum = counts.iloc[0] + counts.iloc[1]
print(counts)
print(round(100*(counts/counts_sum),2))

ind_counts = [v for _, v in sorted(ind_counts.items())]
team_counts = [v for _, v in sorted(team_counts.items())]
chi2, p, dof, expected = stats.chi2_contingency([ind_counts, team_counts])
print(f"Chi-squared Statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies:")
print(expected)
# %%


df = subset[subset['Injury_History'] != "No"]
# df['Injury_History_MoreThanOne (0=no,1=yes)'] = df['Injury_History_MoreThanOne (0=no,1=yes)'].map(lambda x: 1 if x == 1 else 0)

df_ind = df[df['Sports'] == 'individual']

df_team = df[df['Sports'] == 'team']

print(df['Injury_History_MoreThanOne (0=no,1=yes)'].value_counts())
print(df_ind['Injury_History_MoreThanOne (0=no,1=yes)'].value_counts())
print(df_team['Injury_History_MoreThanOne (0=no,1=yes)'].value_counts())

k = 'Injury_History_MoreThanOne (0=no,1=yes)'
d = df
print("\nAll\n")
counts = d[k].value_counts()
counts_sum = counts.iloc[0] + counts.iloc[1]
print(counts)
print(round(100*(counts/counts_sum),2))
print('\nindividual\n')
counts = df_ind[k].value_counts()
ind_counts = counts
counts_sum = counts.iloc[0] + counts.iloc[1]
print(counts)
print(round(100*(counts/counts_sum),2))
print('\nteam\n')
counts = df_team[k].value_counts()
team_counts = counts
counts_sum = counts.iloc[0] + counts.iloc[1]
print(counts)
print(round(100*(counts/counts_sum),2))

ind_counts = [v for _, v in sorted(ind_counts.items())]
team_counts = [v for _, v in sorted(team_counts.items())]

res =  stats.fisher_exact([ind_counts, team_counts])
print(res)

# %%
def binary_question(subset, ind, team, k, method='chi2'):
    d = subset
    print("\nAll\n")
    counts = d[k].value_counts()
    print(counts)
    print(round(100*(counts/counts.sum()),2))
    print('\nindividual\n')
    counts = ind[k].value_counts()
    ind_counts = counts
    print(counts)
    print(round(100*(counts/counts.sum()),2))
    print('\nteam\n')
    counts = team[k].value_counts()
    team_counts = counts
    print(counts)
    print(round(100*(counts/counts.sum()),2))


    ind_counts = [v for _, v in sorted(ind_counts.items())]
    team_counts = [v for _, v in sorted(team_counts.items())]

    if method == 'chi2':
        chi2, p, dof, expected = stats.chi2_contingency([ind_counts, team_counts])
        print(f"Chi-squared Statistic: {chi2}")
        print(f"P-value: {p}")
        print(f"Degrees of Freedom: {dof}")
        print("Expected Frequencies:")
        print(expected)
    elif method == 'fisher':
        res =  stats.fisher_exact([ind_counts, team_counts])
        print(res)

# k = 'Geographic_Factor'
# binary_question(subset, ind, team, k, method='chi2')
# k = 'Hours_per_week>Age'
# binary_question(subset, ind, team, k, method='fisher')
# k = 'Have you given up a sport for your main sport?'
# binary_question(subset, ind, team, k, method='chi2')
# k = 'Is your main sport significantly more important than other sports?'
# binary_question(subset, ind, team, k, method='chi2')
k = 'Months_in_a_year>8'
binary_question(subset, ind, team, k, method='chi2')


# %%
res = stats.levene(ind["Chronologic_Age"], team["Chronologic_Age"])
print(res)
res = stats.levene(ind["Age_started_main_sport"], team["Age_started_main_sport"])
print(res)
res = stats.levene(ind["Experience_main_sport"], team["Experience_main_sport"])
print(res)
res = stats.levene(ind["Training_Volume_Weekly_ALLSports"], team["Training_Volume_Weekly_ALLSports"])
print(res)
res = stats.levene(ind["Training_Volume_Weekly_MainSport"], team["Training_Volume_Weekly_MainSport"])
print(res)

res = stats.kruskal(ind["Chronologic_Age"], team["Chronologic_Age"])


#%%
subset["Sports_numeric"] = subset["Sports"].map({"individual": 0, "team": 1})
df = subset[subset['Injury_History'] == 'Yes']
df = subset
df['low'] = df['Sports_Specialization'].map(lambda x: 1 if x == 'low' else 0)
df['moderate'] = df['Sports_Specialization'].map(lambda x: 1 if x == 'moderate' else 0)
df['high'] = df['Sports_Specialization'].map(lambda x: 1 if x == 'high' else 0)
# Logistic regression for low spec

model_adjusted = smf.logit('Sports_numeric ~  low + Chronologic_Age + Sex + Training_Volume_Weekly_MainSport', data=df).fit()
print(model_adjusted.summary())

# Extract odds ratios and confidence intervals
params = model_adjusted.params
conf = model_adjusted.conf_int()
conf['OR'] = np.exp(params)
conf.columns = ['2.5%', '97.5%', 'OR']


# %%

# Logistic regression model with adjustment
subset["Sports_numeric"] = subset["Sports"].map({"individual": 0, "team": 1})
df = pd.DataFrame(subset)

model_adjusted = smf.logit('Sports_numeric ~  Spec_ordinal + Chronologic_Age + Training_Volume_Weekly_MainSport', data=df).fit()
print(model_adjusted.summary())

# Extract odds ratios and confidence intervals
params = model_adjusted.params
conf = np.exp(model_adjusted.conf_int())
conf['OR'] = np.exp(params)
conf.columns = ['2.5%', '97.5%', 'OR']

print(conf)