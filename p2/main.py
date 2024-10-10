import sys
import os
sys.path.append(os.path.abspath('../')) 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
import utils
from matplotlib_venn import venn3
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

data_HDD = data[columns_to_get + HHD]
data_HDD = process_data(data_HDD)

data_YBT = data[columns_to_get + YBT]
data_YBT = process_data(data_YBT)

data_FMS = data[columns_to_get + FMS]
data_FMS = process_data(data_FMS)

data_HDD_YBT = data[columns_to_get + HHD + YBT]
data_HDD_YBT = process_data(data_HDD_YBT)

data_HDD_FMS = data[columns_to_get + HHD + FMS]
data_HDD_FMS = process_data(data_HDD_FMS)

data_YBT_FMS = data[columns_to_get + YBT + FMS]
data_YBT_FMS = process_data(data_YBT_FMS)

data_HDD_YBT_FMS = data[columns_to_get + HHD + YBT + FMS]
data_HDD_YBT_FMS = process_data(data_HDD_YBT_FMS)

# %%
# Plot Venn Diagram
# The regions are identified via three-letter binary codes ('100', '010', etc), hence a valid artgument could look like:
# A tuple with 7 numbers, denoting the sizes of the regions in the following order:
# (100, 010, 110, 001, 101, 011, 111).
lengths = (
    len(data_HDD),
    len(data_YBT),
    len(data_HDD_YBT),
    len(data_FMS),
    len(data_HDD_FMS),
    len(data_YBT_FMS),
    len(data_HDD_YBT_FMS)
)
venn = venn3(subsets=lengths, set_labels=('HHD', 'YBT', 'FMS'))
plt.title('Liczba "pełnych" sportowców w poszczególnych testach/kombinacjach')
plt.show()


# %%

'''
Hipotezy związane ze Sports Performance (YBT,HHD,FMS)
H1 - Zawodnicy z wysoką specjalizacją (High) będą wykazywać niższe wyniki w Sports Performance Tests
dla dominującej i niedominującej kończyny dolnej (YBT, HHD, FMS)

H2 - Zawodnicy trenujący mniej główny sport („Training_Volume_Weekly_MainSport (hrs)”)
i wszystkie sporty „Training_Volume_Weekly_ALLsports(hrs)” będą wykazywać niższe wyniki
w Sports Performance Tests dla dominującej i niedominującej kończyny dolnej (YBT, HHD, FMS)

H3 - Zawodnicy trenujący więcej niż mają lat (Athletes who participated in their primary sport
for more hours per week than their age (Yes/No) = Yes) będą wykazywać niższe wyniki w Sports Performance Tests
dla dominującej i niedominującej kończyny dolnej (YBT, HHD, FMS)

H4 - Zawodnicy z historią urazów mają niższe wyniki w Sports Performance Tests dla
dominującej i niedominującej kończyny dolnej (YBT, HHD, FMS)

H5 - Zawodnicy z niższymi wartościami jakości zdrowia („QoL - EQ-5D-Y”) wykażą niższe wyniki w
Sports Performance Tests dla dominującej i niedominującej kończyny dolnej (YBT, HHD, FMS)

H6 - Zawodnicy młodsi (Chronologic_Age) i mniej dojrzali biologicznie (maturity_offset)
będą mieli niższe wyniki w Sports Performance Test dla dominującej i niedominującej kończyny dolnej (YBT, HHD, FMS)

----------------------------------------------------------

Jeśli analizujemy zmienne = Sport Performance Tests (YBT, HHD) to
-> należy przypisać kończynę (zmienna = Dominant_extremity) uczestnika do wyników (YBT,HHD)
(czyli jeśli dominant = left, to left YBT i left HHD)

Jeśli analizujemy zmienne = objętość treningowa razem z Injury_History (yes = 1, no = 0)
-> wykluczenie uczestników z „Pain_Now” = 1

'''

# %%

# Create new columns based on the dominant extremity
def assign_dominant_extremity(row, left_col, right_col):
    return row[left_col] if row['Dominant_extremity'] == 'Left' else row[right_col]


data_HDD['PS_DOMINANT_PEAK_FORCE'] = data_HDD.apply(assign_dominant_extremity, left_col='PS_L_PEAK_FORCE', right_col='PS_R_PEAK_FORCE', axis=1)
data_HDD['CZ_DOMINANT_PEAK_FORCE'] = data_HDD.apply(assign_dominant_extremity, left_col='CZ_L_PEAK_FORCE', right_col='CZ_R_PEAK_FORCE', axis=1)
data_HDD['DW_DOMINANT_PEAK_FORCE'] = data_HDD.apply(assign_dominant_extremity, left_col='DW_L_PEAK_FORCE', right_col='DW_R_PEAK_FORCE', axis=1)
data_HDD['BR_DOMINANT_PEAK_FORCE'] = data_HDD.apply(assign_dominant_extremity, left_col='BR_L_PEAK_FORCE', right_col='BR_R_PEAK_FORCE', axis=1)



data_YBT['YBT_ANT_DOMINANT'] = data_YBT.apply(assign_dominant_extremity, left_col='YBT_ANT_L_Normalized', right_col='YBT_ANT_R_Normalized', axis=1)
data_YBT['YBT_PM_DOMINANT'] = data_YBT.apply(assign_dominant_extremity, left_col='YBT_PM_L_Normalized', right_col='YBT_PM_R_Normalized', axis=1)
data_YBT['YBT_PL_DOMINANT'] = data_YBT.apply(assign_dominant_extremity, left_col='YBT_PL_L_Normalized', right_col='YBT_PL_R_Normalized', axis=1)
data_YBT['YBT_COMPOSITE_DOMINANT'] = data_YBT.apply(assign_dominant_extremity, left_col='YBT_COMPOSITE_L', right_col='YBT_COMPOSITE_R', axis=1)

data_YBT['YBT_ANT_NONDOMINANT'] = data_YBT.apply(assign_dominant_extremity, right_col='YBT_ANT_L_Normalized', left_col='YBT_ANT_R_Normalized', axis=1)
data_YBT['YBT_PM_NONDOMINANT'] = data_YBT.apply(assign_dominant_extremity, right_col='YBT_PM_L_Normalized', left_col='YBT_PM_R_Normalized', axis=1)
data_YBT['YBT_PL_NONDOMINANT'] = data_YBT.apply(assign_dominant_extremity, right_col='YBT_PL_L_Normalized', left_col='YBT_PL_R_Normalized', axis=1)
data_YBT['YBT_COMPOSITE_NONDOMINANT'] = data_YBT.apply(assign_dominant_extremity, right_col='YBT_COMPOSITE_L', left_col='YBT_COMPOSITE_R', axis=1)



# %%
# Hypothesis 1
# H1 - Zawodnicy z wysoką specjalizacją (High) będą wykazywać
#  niższe wyniki w Sports Performance Tests
sns.set_theme(style="ticks", palette="pastel")

fig, axes = plt.subplots(4, 1, figsize=(10, 10))

df_melted = data_YBT.melt(id_vars='Sports_Specialization',
                          value_vars=['YBT_ANT_DOMINANT', 'YBT_ANT_NONDOMINANT'],
                          var_name='Dominant Extremity', value_name='Score')

sns.boxplot(ax=axes[0], x='Sports_Specialization', y='Score',
            hue='Dominant Extremity',
            data=df_melted, showfliers=False,
            palette=["m", "g"], order=['low', 'moderate', 'high'])
axes[0].yaxis.set_label_text('YBT ANT')

df_melted = data_YBT.melt(id_vars='Sports_Specialization',
                          value_vars=['YBT_PM_DOMINANT', 'YBT_PM_NONDOMINANT'],
                          var_name='Dominant Extremity', value_name='Score')

sns.boxplot(ax=axes[1], x='Sports_Specialization', y='Score',
            hue='Dominant Extremity',
            data=df_melted, showfliers=False,
            palette=["m", "g"], order=['low', 'moderate', 'high'])
axes[1].yaxis.set_label_text('YBT PM')

df_melted = data_YBT.melt(id_vars='Sports_Specialization',
                          value_vars=['YBT_PL_DOMINANT', 'YBT_PL_NONDOMINANT'],
                          var_name='Dominant Extremity', value_name='Score')

sns.boxplot(ax=axes[2], x='Sports_Specialization', y='Score',
            hue='Dominant Extremity',
            data=df_melted, showfliers=False,
            palette=["m", "g"], order=['low', 'moderate', 'high'])
axes[2].yaxis.set_label_text('YBT PL')

df_melted = data_YBT.melt(id_vars='Sports_Specialization',
                          value_vars=['YBT_COMPOSITE_DOMINANT', 'YBT_COMPOSITE_NONDOMINANT'],
                          var_name='Dominant Extremity', value_name='Score')

sns.boxplot(ax=axes[3], x='Sports_Specialization', y='Score',
            hue='Dominant Extremity',
            data=df_melted, showfliers=False,
            palette=["m", "g"], order=['low', 'moderate', 'high'])
axes[3].yaxis.set_label_text('YBT COMP')

for ax in axes:
    ax.get_legend().remove()
    ax.get_xaxis().get_label().set_visible(False)

# Add a single legend at the bottom of all subplots
handles, _ = axes[0].get_legend_handles_labels()  # Get the handles and labels from one of the plots
labels = ['Dominant', 'Nondominant']
fig.legend(handles, labels, loc='lower center', title= 'Extremity',
           ncol=3, bbox_to_anchor=(0.5, 1), frameon=False)

# Adjust the layout to make room for the legend
plt.tight_layout(rect=[0, 0.03, 1, 1])  # Adjust the bottom space to fit the legend
[ax.set_xticklabels([]) for ax in axes[:-1]]

axes[3].get_xaxis().get_label().set_visible(True)
axes[3].set_xlabel('Sports Specialization')
plt.tight_layout()
plt.show()


# %%


# %%