#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import warnings
import logging
import pandas as pd

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
subset_data = data.iloc[:, 0:25] # 28 with weight height bmi
subset = subset_data.drop(columns=["QoL - EQ-5D-Y",
                                   "Dominant_extremity",])
                                #    "TK TłUSZCZ%"])
# %%
# if we have empty BMI, we can calculate it with formula: weight / (height^2)
# # if we have empty height, we can calculate it with formula: sqrt(weight / BMI) * 100
# # if we have empty weight, we can calculate it with formula: BMI * (height^2)
# subset.loc[subset['BMI'].isna(), 'BMI'] = subset['Weight (kg)'] / ((subset['Height (cm)'] / 100) ** 2)
# subset.loc[subset['Height (cm)'].isna(), 'Height (cm)'] = np.sqrt(subset.loc[subset['Height (cm)'].isna(), 'Weight (kg)'] / subset.loc[subset['Height (cm)'].isna(), 'BMI']) * 100
# subset.loc[subset['Weight (kg)'].isna(), 'Weight (kg)'] = subset.loc[subset['Weight (kg)'].isna(), 'BMI'] * ((subset.loc[subset['Weight (kg)'].isna(), 'Height (cm)'] / 100) ** 2)


# # age phv calculation
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
data = subset


subset = data
data['Sex'] = data['Sex'].map({1: 'Male', 2: 'Female'})
subset = data
ind = data[data["Sports"] == "individual"]
team = data[data["Sports"] == "team"]

# Create a figure and axes with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(14,6), dpi=100)

# Plot 1: Sports Specialization
x, y = 'Sports', 'Sports_Specialization'
df1 = subset.groupby(x)[y].value_counts(normalize=True).mul(100).rename('Percent').reset_index()
sns.barplot(x=x, y='Percent', hue=y, data=df1, hue_order=['low', 'moderate', 'high'], ax=axes[0])
axes[0].set_title('Sports specialization' , fontdict={'fontsize': 14})


# Annotate percentages
ind_spec = ind["Sports_Specialization"].value_counts()
team_spec = team["Sports_Specialization"].value_counts()
items = [ind_spec.low, team_spec.low, ind_spec.moderate, team_spec.moderate, ind_spec.high, team_spec.high]
sums = [ind_spec.sum(), team_spec.sum()]
i = 0
for p in axes[0].patches:
    if i in [0, 2, 4]:
        perc = items[i] / sums[0] * 100
    else:
        perc = items[i] / sums[1] * 100
    axes[0].annotate(f'{perc:.1f}%', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
    i += 1

# Plot 2: Hours per Week vs Age
x, y = 'Sports', 'Hours_per_week>Age'
df2 = subset.groupby(x)[y].value_counts(normalize=True).mul(100).rename('Percent').reset_index()
sns.barplot(x=x, y='Percent', hue=y, data=df2, hue_order=['Yes', 'No'], ax=axes[1])
axes[1].set_title('Exceeding main sport hours:age ratio', fontdict={'fontsize': 14})
axes[1].set_ylabel(None)

# Annotate percentages
ind_more_hrs_than_age = ind["Hours_per_week>Age"].value_counts()
team_more_hrs_than_age = team["Hours_per_week>Age"].value_counts()
items = [ind_more_hrs_than_age.Yes, team_more_hrs_than_age.Yes, ind_more_hrs_than_age.No, team_more_hrs_than_age.No]
sums = [ind_more_hrs_than_age.sum(), team_more_hrs_than_age.sum()]
i = 0
for p in axes[1].patches:
    if i in [0, 2]:
        perc = items[i] / sums[0] * 100
    else:
        perc = items[i] / sums[1] * 100
    axes[1].annotate(f'{perc:.1f}%', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
    i += 1

# Plot 3: Injury History
x, y = 'Sports', 'Injury_History'
df3 = subset.groupby(x)[y].value_counts(normalize=True).mul(100).rename('Percent').reset_index()
g = sns.barplot(x=x, y='Percent', hue=y, data=df3, hue_order=['Yes', 'No'], ax=axes[2])
axes[2].set_title('Injury history', fontdict={'fontsize': 14})

# Annotate percentages
ind_injury_history = ind["Injury_History"].value_counts()
team_injury_history = team["Injury_History"].value_counts()
items = [ind_injury_history.Yes, team_injury_history.Yes, ind_injury_history.No, team_injury_history.No]
sums = [ind_injury_history.sum(), team_injury_history.sum()]
i = 0
for p in axes[2].patches:
    if i in [0, 2]:
        perc = items[i] / sums[0] * 100
    else:
        perc = items[i] / sums[1] * 100
    axes[2].annotate(f'{perc:.1f}%', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
    i += 1

# Adjust layout
def label_iterator():
    yield "a)"
    yield "b)"
    yield "c)"

li = label_iterator()

for ax in axes:
    legend = ax.get_legend()
    legend.set_title(None)
    ax.set_xlabel(next(li), fontdict={'fontsize': 18})
    ax.set_ylabel(None)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

# line1 = plt.Line2D([0.335, 0.335], [0, 1], color="black", linewidth=2, transform=fig.transFigure)
# line2 = plt.Line2D([0.665, 0.665], [0, 1], color="black", linewidth=2, transform=fig.transFigure)
# fig.add_artist(line1)
# fig.add_artist(line2)

plt.tight_layout()
plt.show()
# %%
# Create a figure and axes with 2 rows and 2 columns
fig, axes = plt.subplots(2, 2, figsize=(12,8), dpi=100)

# Plot 1: Age started main sport
sns.boxplot(data=data, x="Age_started_main_sport", y="Sports", width=.6, order=['individual', 'team'], ax=axes[0, 0])
sns.stripplot(data=data, x="Age_started_main_sport", y="Sports", size=4, linewidth=0.3, marker="o", alpha=0.5, color="black", order=['individual', 'team'], ax=axes[0, 0])
axes[0, 0].xaxis.grid(True)
axes[0, 0].set(ylabel="")
sns.despine(trim=True, left=True, ax=axes[0, 0])

g = sns.violinplot(data=subset, x="Age_started_main_sport", y="Sports", hue="Sex", split=True, hue_order=['Male', 'Female'], order=['individual', 'team'], ax=axes[0, 1])
g.set(yticklabels=[])  # remove the tick labels
g.set(xlabel=None)  # remove the axis label

# Plot 2: Training Volume Weekly Main Sport
sns.boxplot(data=data, x="Training_Volume_Weekly_MainSport", y="Sports", width=.6, order=['individual', 'team'], ax=axes[1, 0])
sns.stripplot(data=data, x="Training_Volume_Weekly_MainSport", y="Sports", size=4, linewidth=0.3, marker="o", alpha=0.5, color="black", order=['individual', 'team'], ax=axes[1, 0])
axes[1, 0].xaxis.grid(True)
axes[1, 0].set(ylabel="")
sns.despine(trim=True, left=True, ax=axes[0, 1])

g = sns.violinplot(data=subset, x="Training_Volume_Weekly_MainSport", y="Sports", hue="Sex", split=True, hue_order=['Male', 'Female'], order=['individual', 'team'], ax=axes[1, 1])
g.set(yticklabels=[])  # remove the tick labels
g.set(xlabel=None)  # remove the axis label

axes[0, 0].set_ylabel("a)", rotation=0, fontdict={'fontsize': 18})
axes[0, 1].set_ylabel(None)
axes[1, 0].set_ylabel("b)", rotation=0, fontdict={'fontsize': 18})
axes[1, 1].set_ylabel(None)


axes[0, 0].set_xlabel(None)
axes[0, 1].set_xlabel(None)
axes[1, 0].set_xlabel(None)
axes[1, 1].set_xlabel(None)

axes[0, 0].set_title(None)
axes[0, 1].set_title(None)
axes[1, 0].set_title(None)
axes[1, 1].set_title(None)
# line = plt.Line2D((0, 1), (0.5, 0.5), color="black", linewidth=2, transform=fig.transFigure, figure=fig)
# line2 = plt.Line2D([0.5, 0.5], [0, 1], color="black", linestyle=':', linewidth=1, transform=fig.transFigure)
# fig.add_artist(line)

legend = axes[0, 1].get_legend()
legend.set_title(None)
legend = axes[1, 1].get_legend()
legend.set_title(None)

fig.text(0.5, 0.99, 'Age started main sport', ha='center', va='top', fontsize=16)
fig.text(0.5, 0.49, 'Weekly training volume main sport in hours', ha='center', va='top', fontsize=16)
# Adjust layout
plt.tight_layout()
fig.subplots_adjust(top=0.95, hspace=0.25)  # Adjust hspace to increase/decrease spacing between rows
plt.show()
# %%