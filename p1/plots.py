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
# %%

#%% H1
# H1 – Zawodnicy z indywidualnych sportów („Sports (1 = individual, 2 = team sports)” )
# będą charakteryzować się wyższą (high) specjalizacją niż zawodnicy sportów zespołowych

# sns.countplot(data=data, x="Sports",
#               hue="Sports_Specialization",
#               order=['individual', 'team'],
#               hue_order=['low', 'moderate', 'high'])

x,y = 'Sports', 'Sports_Specialization'
df = subset
df1 = df.groupby(x)[y].value_counts(normalize=True).mul(100).rename('Percent').reset_index()
g = sns.catplot(x=x,y='Percent',hue=y,kind='bar',data=df1, hue_order=['low', 'moderate', 'high'])

ind = data[data["Sports"] == "individual"]
team = data[data["Sports"] == "team"]

ind_spec = ind["Sports_Specialization"].value_counts()
team_spec = team["Sports_Specialization"].value_counts()

items = [ind_spec.low, team_spec.low, ind_spec.moderate, team_spec.moderate, ind_spec.high, team_spec.high]
sums = [ind_spec.sum(), team_spec.sum()]
i = 0
for p in plt.gca().patches:
    if i in [0, 2, 4]:
        perc = items[i] / sums[0] * 100
    else:
        perc = items[i] / sums[1] * 100
    plt.gca().annotate(f'\n{perc:.1f}%', (p.get_x()+p.get_width()/2, p.get_height()), ha='center', va='bottom')
    i += 1
plt.show()
print(ind_spec)
print(team_spec)
#%% H2
# H2 - Zawodnicy z indywidualnych sportów będą charakteryzować się większa liczbą godzin treningu
# niż wiek (Athletes who participated in their primary sport for more hours per week than their age)
# niż zawodnicy sportów zespołowych


# sns.countplot(data=data, x="Sports",
#               hue="Hours_per_week>Age",
#               order=['individual', 'team'],
#               hue_order=['Yes', 'No'])
# for p in plt.gca().patches:
#     plt.gca().annotate(f'\n{int(p.get_height())}', (p.get_x()+p.get_width()/2, p.get_height()), ha='center', va='bottom')
# plt.show()

x,y = 'Sports', 'Hours_per_week>Age'
df = subset
df1 = df.groupby(x)[y].value_counts(normalize=True).mul(100).rename('Percent').reset_index()
g = sns.catplot(x=x,y='Percent',hue=y,kind='bar',data=df1, hue_order=['Yes', 'No'])


ind_more_hrs_than_age = ind["Hours_per_week>Age"].value_counts()
team_more_hrs_than_age = team["Hours_per_week>Age"].value_counts()
items = [ind_more_hrs_than_age.Yes, team_more_hrs_than_age.Yes, ind_more_hrs_than_age.No, team_more_hrs_than_age.No]
sums = [ind_more_hrs_than_age.sum(), team_more_hrs_than_age.sum()]
i = 0
for p in plt.gca().patches:
    if i in [0, 2]:
        perc = items[i] / sums[0] * 100
    else:
        perc = items[i] / sums[1] * 100
    plt.gca().annotate(f'\n{perc:.1f}%', (p.get_x()+p.get_width()/2, p.get_height()), ha='center', va='bottom')
    i += 1

print(f'{ind_more_hrs_than_age=}')
print(f'{team_more_hrs_than_age=}')

#%% H4
# H4 – Zawodnicy z indywidualnych sportów będą charakteryzować się
# większą liczbą zawodników z urazami w przeszłości („Injury_History (yes = 1, no = 0)’ historii urazów)
# niż zawodnicy sportów zespołowych

# data_withouth_na = data.dropna(subset=["Injury_History"])


# sns.countplot(data=data, x="Sports",
#               hue="Injury_History", order=['individual', 'team'], hue_order=['Yes', 'No'])

x,y = 'Sports', 'Injury_History'
df = subset
df1 = df.groupby(x)[y].value_counts(normalize=True).mul(100).rename('Percent').reset_index()
g = sns.catplot(x=x,y='Percent',hue=y,kind='bar',data=df1, hue_order=['Yes', 'No'])

ind_injury_history= ind["Injury_History"].value_counts()
team_injury_history = team["Injury_History"].value_counts()
items = [ind_injury_history.Yes, team_injury_history.Yes, ind_injury_history.No, team_injury_history.No]
sums = [ind_injury_history.sum(), team_injury_history.sum()]
i = 0
for p in plt.gca().patches:
    if i in [0, 2]:
        perc = items[i] / sums[0] * 100
    else:
        perc = items[i] / sums[1] * 100
    plt.gca().annotate(f'\n{perc:.1f}%', (p.get_x()+p.get_width()/2, p.get_height()), ha='center', va='bottom')
    i += 1

print(f'{ind_more_hrs_than_age=}')
print(f'{team_more_hrs_than_age=}')
#%% H5
# H5 - Zawodnicy trenujący więcej niż 8 miesiecy w roku jeden główny sport  „Have you trained in a main sport for more than 8 months in one year?”
# będą mieli więcej historii urazów („Injury_History (yes = 1, no = 0)

# sns.countplot(data=data, x="Months_in_a_year>8",
#               hue="Injury_History", order=['Yes', 'No'], hue_order=['Yes', 'No'])

x,y = 'Months_in_a_year>8', 'Injury_History'
df = subset
df1 = df.groupby(x)[y].value_counts(normalize=True).mul(100).rename('Percent').reset_index()
g = sns.catplot(x=x,y='Percent',hue=y,kind='bar',data=df1, hue_order=['Yes', 'No'], order=['Yes', 'No'])
plt.xlabel("Trains main sport for more than 8 months in one year")
more_than_8_months_in_year = data[data["Months_in_a_year>8"] == "Yes"]["Injury_History"].value_counts()
less_than_8_months_in_year = data[data["Months_in_a_year>8"] == "No"]["Injury_History"].value_counts()

items = [more_than_8_months_in_year.Yes, less_than_8_months_in_year.Yes, more_than_8_months_in_year.No, less_than_8_months_in_year.No]
sums = [more_than_8_months_in_year.sum(), less_than_8_months_in_year.sum()]
i = 0
for p in plt.gca().patches:
    if i in [0, 2]:
        perc = items[i] / sums[0] * 100
    else:
        perc = items[i] / sums[1] * 100
    plt.gca().annotate(f'\n{perc:.2f}%', (p.get_x()+p.get_width()/2, p.get_height()), ha='center', va='bottom')
    i += 1
#%% H3
# H3 Zawodnicy z indywidualnych sportów będą charakteryzować się
# większa objętością treningową „Training_Volume_Weekly_MainSport (hrs)”
# w porównaniu do zawodników sportów zespołowych


# sns.displot(data=data, x="Training_Volume_Weekly_MainSport (hrs)",
#             hue="Sports", multiple="stack", kind="kde")

f, ax = plt.subplots(figsize=(7, 6))

sns.boxplot(data=data, x="Training_Volume_Weekly_MainSport", y="Sports",
            width=.6, order=['individual', 'team'])
sns.stripplot(data, x="Training_Volume_Weekly_MainSport", y="Sports",
              size=4, linewidth=0.3,
              marker="o", alpha=0.5, color="black", order=['individual', 'team'])

ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
plt.xlabel("Training Volume Weekly Main Sport (hrs)")
plt.show()

print(ind["Training_Volume_Weekly_MainSport"].describe())
print(team["Training_Volume_Weekly_MainSport"].describe())


data[' Sex '] = data['Sex'].map({1: 'M', 2: 'F'})
sns.catplot(
    data=df, x="Training_Volume_Weekly_MainSport", y="Sports", hue=" Sex ",
    kind="violin", split=True, hue_order=['M', 'F'], order=['individual', 'team'],
)
plt.xlabel("Training Volume Weekly Main Sport (hrs)")
plt.show()

sns.catplot(
    data=df, x="Training_Volume_Weekly_MainSport", y="Sports_Specialization", hue="Sports",
    kind="violin", split=True, hue_order=['individual', 'team'], order=['low', 'moderate', 'high'],
)
plt.xlabel("Training Volume Weekly Main Sport (hrs)")
plt.ylabel("Sports Specialization")
plt.show()
#%% H6
# H6 – Różnica między grupami (Sports (1 = individual, 2 = team sports)”) i lokalizacją i typ urazu (Injury  Location (Lower or Upper)
# and Type (Acute vs Overuse) „Injury_History_Localization_Upper or Lower Limbs”,
# „Injury_History_Overuse_Acute” (differences between the injury type (acute versus overuse) among team sport and individual sport athletes)
fig, axes = plt.subplots(1, 2, figsize=(14,6), dpi=100)


ind_injury = ind["Injury_History_Localization_Upper_Lower_Torso"] + ind["Injury_History_Overuse_Acute"]
team_injury = team["Injury_History_Localization_Upper_Lower_Torso"] + team["Injury_History_Overuse_Acute"]

ind_injury = ind_injury.value_counts()
team_injury = team_injury.value_counts()
labels = [f"Upper Overuse\n{0}", f"Upper Acute\n{team_injury.UpperAcute}",
          f"Torso Acute\n{team_injury.TorsoAcute}", f"Lower Acute\n{team_injury.LowerAcute}",
          f"Lower Overuse\n{team_injury.LowerOveruse}", f"Torso Overuse\n{team_injury.TorsoOveruse}"]
print(f'{team_injury=}')
print(f'{ind_injury=}')

ind_injury = ind_injury / ind_injury.max()
team_injury = team_injury / team_injury.max()

if "UpperOveruse" not in ind_injury.keys():
    ind_injury.UpperOveruse = 0

if "UpperOveruse" not in team_injury.keys():
    team_injury.UpperOveruse = 0
proportions = [team_injury.UpperOveruse, team_injury.UpperAcute,
               team_injury.TorsoAcute, team_injury.LowerAcute,
               team_injury.LowerOveruse, team_injury.TorsoOveruse]


proportions_ind = [ind_injury.UpperOveruse, ind_injury.UpperAcute,
                     ind_injury.TorsoAcute, ind_injury.LowerAcute,
                     ind_injury.LowerOveruse, ind_injury.TorsoOveruse]


N = len(proportions)
proportions /= np.array(proportions).max()
proportions = np.append(proportions, 1)
print(proportions)
proportions_ind /= np.array(proportions_ind).max()
proportions_ind = np.append(proportions_ind, 1)
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
x = np.append(np.sin(theta-2*np.pi/12), 0)
y = np.append(np.cos(theta-2*np.pi/12), 0)
triangles = [[N, i, (i + 1) % N] for i in range(N)]
# triangles[1][0] = 0
triang_backgr = tri.Triangulation(x, y, triangles)
triang_foregr = tri.Triangulation(x * proportions, y * proportions, triangles)
triang_foregr_ind = tri.Triangulation(x * proportions_ind, y * proportions_ind, triangles)

colors = np.array([0.1]*N)
colors_ind = np.array([0.]*N)
c = np.array([0] * N)
axes[0].tripcolor(triang_backgr, c , cmap='rainbow', alpha=0.1)
# plt.tripcolor(triang_foregr_ind, colors_ind, cmap='autumn', alpha=1)
axes[0].tripcolor(triang_foregr, colors, cmap='bwr', alpha=.5, edgecolors='k')
# leg = plt.legend(['team', 'individual'], loc='best',
#                  bbox_to_anchor=(0., 0., 0.1, 0.9), fontsize='small', title='Sports')
color = matplotlib.colormaps['autumn'](0)
# leg.legend_handles[0].set_color(color)
color = matplotlib.colormaps['bwr'](0)
# leg.legend_handles[1].set_color(color)
axes[0].triplot(triang_backgr, 'o:', color='black', lw=1.5, alpha=0.5)
# plt.triplot(triang_foregr_ind, '-', color='black', lw=.5)
# plt.triplot(triang_foregr, '-', color='black', lw=.5)


for label, color, xi, yi in zip(labels, colors, x, y):
    axes[0].text(xi * 1.05, yi * 1.05, label,  # color=cmap(color),
             ha='left' if xi > 0.1 else 'right' if xi < -0.1 else 'center',
             va='bottom' if yi > 0.1 else 'top' if yi < -0.1 else 'center')
axes[0].axis('off')
plt.gca().set_aspect('equal')



labels = [f"Upper Overuse\n{0}", f"Upper Acute\n{ind_injury.UpperAcute}",
          f"Torso Acute\n{ind_injury.TorsoAcute}", f"Lower Acute\n{ind_injury.LowerAcute}",
          f"Lower Overuse\n{ind_injury.LowerOveruse}", f"Torso Overuse\n{ind_injury.TorsoOveruse}"]
ind_injury = ind_injury / ind_injury.max()
team_injury = team_injury / team_injury.max()

if "UpperOveruse" not in ind_injury.keys():
    ind_injury.UpperOveruse = 0

if "UpperOveruse" not in team_injury.keys():
    team_injury.UpperOveruse = 0
proportions = [team_injury.UpperOveruse, team_injury.UpperAcute,
               team_injury.TorsoAcute, team_injury.LowerAcute,
               team_injury.LowerOveruse, team_injury.TorsoOveruse]


proportions_ind = [ind_injury.UpperOveruse, ind_injury.UpperAcute,
                     ind_injury.TorsoAcute, ind_injury.LowerAcute,
                     ind_injury.LowerOveruse, ind_injury.TorsoOveruse]


N = len(proportions)
proportions /= np.array(proportions).max()
proportions = np.append(proportions, 1)
print(proportions)
proportions_ind /= np.array(proportions_ind).max()
proportions_ind = np.append(proportions_ind, 1)
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
x = np.append(np.sin(theta-2*np.pi/12), 0)
y = np.append(np.cos(theta-2*np.pi/12), 0)
triangles = [[N, i, (i + 1) % N] for i in range(N)]
# triangles[1][0] = 0
triang_backgr = tri.Triangulation(x, y, triangles)
triang_foregr = tri.Triangulation(x * proportions, y * proportions, triangles)
triang_foregr_ind = tri.Triangulation(x * proportions_ind, y * proportions_ind, triangles)

colors = np.array([0.1]*N)
colors_ind = np.array([0.]*N)
c = np.array([0] * N)
# axes[1].tripcolor(triang_backgr, c , cmap='rainbow', alpha=0.1)
axes[1].tripcolor(triang_foregr_ind, colors_ind, cmap='autumn', alpha=1)
# axes[1].tripcolor(triang_foregr, colors, cmap='bwr', alpha=.5, edgecolors='k')
# leg = plt.legend(['team', 'individual'], loc='best',
#                  bbox_to_anchor=(0., 0., 0.1, 0.9), fontsize='small', title='Sports')
color = matplotlib.colormaps['autumn'](0)
# leg.legend_handles[0].set_color(color)
color = matplotlib.colormaps['bwr'](0)
# leg.legend_handles[1].set_color(color)
axes[1].triplot(triang_backgr, 'o:', color='black', lw=1.5, alpha=0.5)
# plt.triplot(triang_foregr_ind, '-', color='black', lw=.5)
# plt.triplot(triang_foregr, '-', color='black', lw=.5)


fig.text(0.5, 0.99, 'Injury localization&type', ha='center', va='top', fontsize=16)
for label, color, xi, yi in zip(labels, colors, x, y):
    axes[1].text(xi * 1.05, yi * 1.05, label,  # color=cmap(color),
             ha='left' if xi > 0.1 else 'right' if xi < -0.1 else 'center',
             va='bottom' if yi > 0.1 else 'top' if yi < -0.1 else 'center')
axes[1].axis('off')
plt.gca().set_aspect('equal')

plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from math import pi

categories = ['Torso\nAcute', 'Upper\nAcute', 'Upper\nOveruse', 'Torso\nOveruse', 'Lower\nOveruse', 'Lower\nAcute']
num_vars = len(categories)

team_sports = [4, 15, 0, 2, 33, 31]
individual_sports = [3, 8, 0, 2, 8, 8]

def normalize_data(data, target_sum):
    current_sum = sum(data)
    scale_factor = target_sum / current_sum
    return [x * scale_factor for x in data]

target_sum = 1
team_sports_normalized = normalize_data(team_sports, target_sum)
individual_sports_normalized = normalize_data(individual_sports, target_sum)

angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]

def plot_radar_chart(title, labels, data, color):
    data += data[:1]  # Ensure the plot is closed
    ax.fill(angles, data, color=color, alpha=0.25)
    ax.plot(angles, data, color=color, linewidth=2, linestyle='solid', label=title)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

plot_radar_chart('team', categories, team_sports_normalized, 'red')

plot_radar_chart('individual', categories, individual_sports_normalized, 'blue')

plt.xticks(angles[:-1], categories, color='grey', size=14)
ax.tick_params(axis='x', which='major', pad=20)
# Add yticks and customize them

ax.set_yticks([.2/2, .4/2, .6/2, .8/2, 1./2])
ax.set_yticklabels(['10%', '20%', '30%', '40%', '50%'])
ax.set_rlabel_position(160)
plt.title('Normalized Sports Injury Data', size=15, color='darkblue', y=1.1)

plt.legend(loc='upper left', bbox_to_anchor=(-0.1, 1.1), fontsize=12)

ax.spines['polar'].set_visible(True)  # Show the frame
ax.grid(color='grey', linestyle='--', linewidth=0.5)
ax.set_facecolor('#f7f7f7')  # Light grey background

plt.show()


# %%
ind_injury = ind["Injury_History_Localization_Upper_Lower_Torso"] + ind["Injury_History_Overuse_Acute"]
team_injury = team["Injury_History_Localization_Upper_Lower_Torso"] + team["Injury_History_Overuse_Acute"]

ind_injury = ind_injury.value_counts()
team_injury = team_injury.value_counts()

if "UpperOveruse" not in team_injury.keys():
    team_injury.UpperOveruse = 0

proportions = [team_injury.TorsoAcute, team_injury.UpperAcute,
               team_injury.UpperOveruse, team_injury.TorsoOveruse,
               team_injury.LowerOveruse, team_injury.LowerAcute]

if "UpperOveruse" not in ind_injury.keys():
    ind_injury.UpperOveruse = 0

proportions_ind = [ind_injury.TorsoAcute, ind_injury.UpperAcute,
                   ind_injury.UpperOveruse, ind_injury.TorsoOveruse,
                   ind_injury.LowerOveruse, ind_injury.LowerAcute]

categories= ['TorsoAcute', 'UpperAcute', 'UpperOveruse', 'TorsoOveruse', 'LowerOveruse', 'LowerAcute']
N = len(categories)

values = proportions
values += values[:1]
values
 
# angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

ax = plt.subplot(111, polar=True)

plt.xticks(angles[:-1], categories, color='grey', size=15)
ax.set_rlabel_position(5)
plt.yticks([10,20,30,40], ["10","20","30", "40"], color="grey", size=8)
plt.ylim(0,40)
ax.plot(angles, values, linewidth=1, linestyle='solid')
ax.fill(angles, values, 'b', alpha=0.1, label='_nolegend_')

values = proportions_ind
values += values[:1]
values

ax.plot(angles, values, linewidth=1, linestyle='solid')
ax.fill(angles, values, 'r', alpha=0.1, label='_nolegend_')
ax.spines['end'].set_visible(False)
ax.spines['start'].set_visible(False)
ax.spines['polar'].set_visible(False)
ax.spines['inner'].set_visible(False)

leg = plt.legend(['team', 'individual'], loc='best',
                 bbox_to_anchor=(0., 0., 0.05, 0.9), fontsize='small', title='Sports')
plt.show()



# %% H7
# Hipoteza = Zawodnicy ze sportów indywidualnych wcześniej zaczeli główny sport
# niż zawodnicy ze sportów drużynowych.


f, ax = plt.subplots(figsize=(7, 6))

sns.boxplot(data=data, x="Age_started_main_sport", y="Sports",
            width=.6, order=['individual', 'team'])
sns.stripplot(data, x="Age_started_main_sport", y="Sports",
              size=4, linewidth=0.3,
              marker="o", alpha=0.5, color="black", order=['individual', 'team'])

ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
plt.xlabel("Age started main sport")
plt.show()

print(ind["Age_started_main_sport"].describe())
print(team["Age_started_main_sport"].describe())
# %%
data[' Sex '] = data['Sex'].map({1: 'M', 2: 'F'})
sns.catplot(
    data=df, x="Age_started_main_sport", y="Sports", hue=" Sex ",
    kind="violin", split=True, hue_order=['M', 'F'], order=['individual', 'team'],
)
plt.xlabel("Age started main sport")
plt.show()


# %%
# Figura. Distribution of single-sport specialized atheltes (train >8 months/year for main sport)
# among reported individual and team sport
sports_colors = {
    'chess': '#1f77b4',      # blue
    'moto-cross': '#ff7f0e',  # orange
    'gimnastics': '#2ca02c',    # green
    'soccer': '#d62728',      # red
    'swimming': '#9467bd',      # purple
    'taekwondo': '#8c564b',    # brown
    'rugby': '#e377c2',     # pink
    'basketball': '#7f7f7f',     # grey
    'handball': '#bcbd22',      # yellow
    'karate': '#17becf', # cyan
    'roller skating': '#ff0000', # red
    'table tennis': '#00ff00', # green
    'sailing': '#0000ff', # blue
    'pole vault': '#ff00ff', # magenta  
}

fig, ax = plt.subplots(figsize=(8, 5))
d = ind[ind["Months_in_a_year>8"] == "Yes"]
b = ind[ind["Months_in_a_year>8"] == "No"]

g = sns.countplot(data=d, y="Sport", order=d["Sport"].value_counts().index, palette=sports_colors,)
ax.set(ylabel="")
plt.title("Individual sport athletes who train >8 months/year for main sport")
plt.show()



fig, ax = plt.subplots(figsize=(8, 2.5))
d = team[team["Months_in_a_year>8"] == "Yes"]
g = sns.countplot(data=team, y="Sport", order=d["Sport"].value_counts().index,
                  width=.6)
ax.set(ylabel="")
plt.title("Team sport athletes who train >8 months/year for main sport")
plt.show()

# %%
# Figure. Average age at which the single-sport specialized atheltes (train >8 months/year)
# started specializing in their sport. Compariosn cross the 10 most commonly reported sports.

fig, ax = plt.subplots(figsize=(8, 5))
d = subset[subset['Months_in_a_year>8'] == "Yes"]
top_10_sports = d["Sport"].value_counts().nlargest(10).index
data_top_10 = d[subset["Sport"].isin(top_10_sports)]

order = data_top_10.groupby("Sport")["Age_started_main_sport"].mean().sort_values(ascending=True).index
# order = data_top_10["Sport"].value_counts(normalize=True).index
sns.boxplot(data=data_top_10, x="Age_started_main_sport", y="Sport",
            width=.6, order=order, linewidth=0.5, palette=sports_colors)
sns.stripplot(data=data_top_10, x="Age_started_main_sport", y="Sport",
              size=4, linewidth=0.3,
              marker="o", alpha=0.5, color="black", order=order) #top_10_sports)
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
plt.title("Average age at which the single-sport specialized (who train >8 months/year for main sport)\nathletes started specializing in their sport")
plt.xlabel("Age started main sport")
plt.show()


# %%
g = sns.catplot(data=data_top_10, x="Sport",
                y="Age_started_main_sport",
                kind='bar', errorbar=('sd'), capsize=0.2, aspect=1.5,
                palette=sports_colors,
                order=data_top_10.groupby("Sport")["Age_started_main_sport"].mean().sort_values(ascending=True).index)
g.set_xticklabels(rotation=55)
plt.ylabel("Age started main sport")


tic = data_top_10.groupby("Sport")["Age_started_main_sport"].mean().sort_values(ascending=True)

tic2 = data_top_10['Sport'].value_counts()[tic.index]
tic2 = [f'{tic2.index[item]} (n={tic2.iloc[item]})' for item in range(len(tic2))]

plt.xticks(range(len(data_top_10["Sport"].value_counts())), tic2)

ax = g.axes[0][0]
ticks = ax.get_xticks()
labels = ax.get_xticklabels()
shift = -0.4
new_ticks = [tick + shift for tick in ticks]
ax.set_xticks(new_ticks)
ax.set_xticklabels(labels)

plt.xlabel("Sport (n={})".format(len(data_top_10)))
plt.title("Average age at which the single-sport specialized (who train >8 months/year for main sport)\nathletes started specializing in their sport (MEAN ± SD)")
a = data_top_10['Sport'].value_counts()


# %%
# g = sns.pairplot(data=subset_data, hue="Dominant_extremity")
# for ax in g.axes.flatten():
#     # rotate x axis labels
#     ax.set_xlabel(ax.get_xlabel(), rotation = 90)
#     # rotate y axis labels
#     ax.set_ylabel(ax.get_ylabel(), rotation = 0)
#     # set y labels alignment
#     ax.yaxis.get_label().set_horizontalalignment('right')
# plt.show()
# %%
mean = data_top_10.groupby("Sport")["Training_Volume_Weekly_ALLSports"].mean()
std = data_top_10.groupby("Sport")["Training_Volume_Weekly_ALLSports"].std()

for sport, m, s in zip(mean.index, mean, std):
    print(f"{sport}: {m:.2f}±{s:.2f}")
# %%
data_top_10.groupby("Sport")['Geographic_Factor'].value_counts()
data_top_10.groupby("Sport")['Geographic_Factor'].value_counts(normalize=True).mul(100).rename('Percent').reset_index()

