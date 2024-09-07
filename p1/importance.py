#%%
import os
import numpy as np
import sklearn
import shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import scipy.stats as stats
import logging
import pandas as pd
import statsmodels.api as sm
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
# if we have empty weight, we can calculate it with formula: BMI * (height^2)
# if we have empty height, we can calculate it with formula: sqrt(weight / BMI) * 100
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



subset["Geographic_Factor"] = subset["Geographic_Factor"].map({"Urban": 0, "Rural": 1})
subset["Sports_numeric"] = subset["Sports"].map({"individual": 0, "team": 1})
subset["Hours_per_week>Age"] = subset["Hours_per_week>Age"].map({"No": 0, "Yes": 1})
subset["Have you given up a sport for your main sport?"] = subset["Have you given up a sport for your main sport?"].map({"No": 0, "Yes": 1})
subset["Is your main sport significantly more important than other sports?"] = subset["Is your main sport significantly more important than other sports?"].map({"No": 0, "Yes": 1})
subset["Months_in_a_year>8"] = subset["Months_in_a_year>8"].map({"No": 0, "Yes": 1})
subset["Sex"] = subset["Sex"].map({1: 1, 2: 0})
subset["Sports_numeric"] = subset["Sports"].map({"individual": 0, "team": 1})
subset["Injury_History"] = subset["Injury_History"].map({"No": 0, "Yes": 1})
subset["Injury_History_MoreThanOne (0=no,1=yes)"] = subset["Injury_History_MoreThanOne (0=no,1=yes)"].map({"": 0, "1":1})
subset["Injury_History_Overuse_Acute"] = subset["Injury_History_Overuse_Acute"].map({"": 0, "Overuse": 1, "Acute": 2})
subset["Injury_History_Localization_Upper_Lower_Torso"] = subset["Injury_History_Localization_Upper_Lower_Torso"].map({"": 0, "Upper": 1, "Lower": 2, "Torso": 3})

dependent_vars = ['Sports', 'Sport', 'Sports_numeric', 'Injury_History_Localization', 'ID', 'Lp', 'Sports_Specialization', 'Pain_now (0=NO, 1=YES)']
independent_vars = subset.columns.difference(dependent_vars)

# %%
# Split the DataFrame
X = subset[independent_vars]  # Independent variables
y = subset['Sports_numeric']    # Dependent variables




X = X.rename(columns={"Chronologic_Age": "Chronologic age",
                      "Experience_main_sport": "Experience main sport",
                      "Training_Volume_Weekly_MainSport": "Weekly training volume main sport",
                      "Training_Volume_Weekly_ALLSports": "Weekly training volume all sports",
                      "Age_started_main_sport": "Age started main sport",
                      "Geographic_Factor": "Geographic factor",
                      "Sports_Specialization (points) (0-1 low, 2 moderate, 3 high)'": "Sports specialization",
                      "Spec_ordinal": "Sports specialization ordinal",
                      "Hours_per_week>Age": "Exceeding main sport hours:age ratio",
                      "Injury_History": "Injury history",
                      "Injury_History_MoreThanOne (0=no,1=yes)": "Multiple injuries",
                      "Injury_History_Overuse_Acute": "Injury type",
                      "Injury_History_Localization_Upper_Lower_Torso": "Injury localization",
                      })




import scipy.stats as stats
# Perform ANOVA for the first outcome
anova_results_y1 = {col: stats.f_oneway(X[col], y) for col in X.columns}
# Perform ANOVA for the second outcome
# Combine p-values
# Sort by combined p-values
sorted_p_values = pd.Series(anova_results_y1).sort_values()
# Select the top 2 variables
top_variables = sorted_p_values.head(2)
print(sorted_p_values)
# %%


# a classic housing price dataset

# a simple linear model
model = sklearn.linear_model.LogisticRegression()
model.fit(X, y)
# %%
print("Model coefficients:\n")
model.coef_
# %%
shap.partial_dependence_plot(
    "Chronologic age",
    model.predict,
    X,
    ice=False,
    model_expected_value=True,
    feature_expected_value=True,
)
# %%
explainer = shap.Explainer(model.predict, X)
shap_values = explainer(X)

# %%
shap.plots.bar(shap_values.abs.max(0))
# %%
def model_log_odds(x):
    p = model.predict_log_proba(x)
    return p[:, 1] - p[:, 0]

def model_proba(x):
    return model.predict_proba(x)[:, 1]

sample_ind = 18
fig, ax = shap.partial_dependence_plot(
    "Chronologic age",
    model_proba,
    X,
    model_expected_value=True,
    feature_expected_value=True,
    show=False,
    ice=False,
)

# compute the SHAP values for the linear model
explainer = shap.Explainer(model_proba, X)
shap_values = explainer(X)

shap.plots.scatter(shap_values[:, "Chronologic age"])

## %%
# compute the SHAP values for the linear model
explainer_log_odds = shap.Explainer(model_log_odds, X)
shap_values_log_odds = explainer_log_odds(X)
shap.plots.scatter(shap_values_log_odds[:, "Chronologic age"])
shap.plots.bar(shap_values_log_odds)
shap.plots.bar(shap_values_log_odds.abs.max(0))

# %%
clustering = shap.utils.hclust(X, y)
shap.plots.bar(shap_values_log_odds, clustering=clustering)

# %%
plt.ion()
fig, ax = plt.subplots()
f = shap.heatmap_plot(shap_values_log_odds, show=False, ax=ax)

children = ax.get_children()
for i, child in enumerate(children):
    print(f'Child {i}: {type(child)}')
    if type(child) == plt.Text:
        print(child.get_text())
# f_x_element = children[1]
# f_x_element.remove()
f_x_element = children[2]
f_x_element.remove()

# Display the updated plot without the f(x) element
plt.show()
# %%
