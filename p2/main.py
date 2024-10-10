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

data_HDD['PS_NONDOMINANT_PEAK_FORCE'] = data_HDD.apply(assign_dominant_extremity, right_col='PS_L_PEAK_FORCE', left_col='PS_R_PEAK_FORCE', axis=1)
data_HDD['CZ_NONDOMINANT_PEAK_FORCE'] = data_HDD.apply(assign_dominant_extremity, right_col='CZ_L_PEAK_FORCE', left_col='CZ_R_PEAK_FORCE', axis=1)
data_HDD['DW_NONDOMINANT_PEAK_FORCE'] = data_HDD.apply(assign_dominant_extremity, right_col='DW_L_PEAK_FORCE', left_col='DW_R_PEAK_FORCE', axis=1)
data_HDD['BR_NONDOMINANT_PEAK_FORCE'] = data_HDD.apply(assign_dominant_extremity, right_col='BR_L_PEAK_FORCE', left_col='BR_R_PEAK_FORCE', axis=1)


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

def create_boxplot_4row(data, id_var, value_vars_list, row_labels, hue_label, legend_labels, palette, order, x_label):
    sns.set_theme(style="ticks", palette="pastel")

    fig, axes = plt.subplots(4, 1, figsize=(10, 10))

    for i, value_vars in enumerate(value_vars_list):
        df_melted = data.melt(id_vars=id_var,
                              value_vars=value_vars,
                              var_name=hue_label, value_name='Score')

        sns.boxplot(ax=axes[i], x=id_var, y='Score',
                    hue=hue_label,
                    data=df_melted, showfliers=False,
                    palette=palette, order=order)
        axes[i].yaxis.set_label_text(row_labels[i])

    for ax in axes:
        ax.get_legend().remove()
        ax.get_xaxis().get_label().set_visible(False)
        ax.grid(axis='y')

    handles, _ = axes[0].get_legend_handles_labels()  # Get handles and labels from one of the plots
    fig.legend(handles, legend_labels, loc='lower center', title=hue_label,
               ncol=3, bbox_to_anchor=(0.5, 1), frameon=False)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    [ax.set_xticklabels([]) for ax in axes[:-1]]

    axes[3].get_xaxis().get_label().set_visible(True)
    axes[3].set_xlabel(x_label)
    plt.tight_layout()
    plt.show()

create_boxplot_4row(
    data=data_YBT, 
    id_var='Sports_Specialization', 
    value_vars_list=[
        ['YBT_ANT_DOMINANT', 'YBT_ANT_NONDOMINANT'],
        ['YBT_PM_DOMINANT', 'YBT_PM_NONDOMINANT'],
        ['YBT_PL_DOMINANT', 'YBT_PL_NONDOMINANT'],
        ['YBT_COMPOSITE_DOMINANT', 'YBT_COMPOSITE_NONDOMINANT']
    ], 
    row_labels=['YBT ANT', 'YBT PM', 'YBT PL', 'YBT COMP'], 
    hue_label='Dominant Extremity', 
    legend_labels=['Dominant', 'Nondominant'], 
    palette=["m", "g"], 
    order=['low', 'moderate', 'high'], 
    x_label='Sports Specialization'
)
create_boxplot_4row(
    data=data_HDD, 
    id_var='Sports_Specialization', 
    value_vars_list=[
        ['PS_DOMINANT_PEAK_FORCE', 'PS_NONDOMINANT_PEAK_FORCE'],
        ['CZ_DOMINANT_PEAK_FORCE', 'CZ_NONDOMINANT_PEAK_FORCE'],
        ['DW_DOMINANT_PEAK_FORCE', 'DW_NONDOMINANT_PEAK_FORCE'],
        ['BR_DOMINANT_PEAK_FORCE', 'BR_NONDOMINANT_PEAK_FORCE']
    ], 
    row_labels=['HDD PS', 'HDD CZ', 'HDD DW', 'HDD BR'], 
    hue_label='Dominant Extremity', 
    legend_labels=['Dominant', 'Nondominant'], 
    palette=["m", "g"], 
    order=['low', 'moderate', 'high'], 
    x_label='Sports Specialization'
)

# %%

sns.boxplot(x='Sports_Specialization', y='FMS_TOTAL', data=data_FMS,
            order=['low', 'moderate', 'high'])
sns.stripplot(x='Sports_Specialization', y='FMS_TOTAL', data=data_FMS, color=".3")
plt.xlabel('Sports Specialization')
plt.ylabel('FMS Total Score')
plt.grid(axis='y')
plt.show()



# %%
# Hypothesis 2
# H2 - Zawodnicy trenujący mniej główny sport („Training_Volume_Weekly_MainSport (hrs)”)
# i wszystkie sporty „Training_Volume_Weekly_ALLsports(hrs)” będą wykazywać niższe wyniki
# w Sports Performance Tests dla dominującej i niedominującej kończyny dolnej (YBT, HHD, FMS)


def ybt_box_plot(data, x, y1, y2, title1, title2, xlabel, ylabel):
    data_YBT = data
# Cut dataframe into 4 equal bins based on x and name the ranges with numeric stats
    bins = pd.cut(data_YBT[x], bins=4, include_lowest=True)
    bin_labels = [f"{round(b.left, 1)}-{round(b.right, 1)}" for b in bins.cat.categories]
    bin_labels[0] = str(min(data_YBT[x])) + bin_labels[0][3:]
    #\2013 for n-dash (number ranges), \u2014 for m-dash
    bin_labels = [f"{b}".replace('-', '\u2013') for b in bin_labels]

    data_YBT['Training_Volume_Binned'] = pd.cut(data_YBT[x], bins=4,
                                            labels=bin_labels, include_lowest=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    sns.boxplot(x='Training_Volume_Binned', y=y1, data=data_YBT, ax=axes[0])
    sns.stripplot(x='Training_Volume_Binned', y=y1, data=data_YBT, ax=axes[0], color=".3")
    axes[0].set_title(title1)
    axes[0].set_ylabel(ylabel)

    sns.boxplot(x='Training_Volume_Binned', y=y2, data=data_YBT, ax=axes[1])
    sns.stripplot(x='Training_Volume_Binned', y=y2, data=data_YBT, ax=axes[1], color=".3")
    axes[1].set_title(title2)
    axes[1].set_ylabel(ylabel)
    ylim_ax0 = axes[0].get_ylim()
    ylim_ax1 = axes[1].get_ylim()
    upper_lim = math.ceil(max(ylim_ax0[1], ylim_ax1[1]))
    lower_lim = math.floor(min(ylim_ax0[0], ylim_ax1[0]))


    # Remove individual x-axis labels
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    
    # Set a shared x-axis label for the whole row
    fig.supxlabel(xlabel)

    for ax in axes:
        ax.set_ylim([lower_lim, upper_lim])
        ax.grid(axis='y')

    plt.tight_layout()
    plt.show()

title1 = 'Dominant Extremity'
title2 = 'Non Dominant Extremity'

x = 'Training_Volume_Weekly_ALLSports'
xlabel = 'Training Volume All Sports [hrs/week]'

# x = 'Training_Volume_Weekly_MainSport'
# xlabel = 'Training Volume Main Sport [hrs/week]'

ybt_box_plot(data_YBT, x=x,
             y1='YBT_ANT_DOMINANT',
             y2='YBT_ANT_NONDOMINANT',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='YBT ANT')
ybt_box_plot(data_YBT, x=x,
             y1='YBT_PM_DOMINANT',
             y2='YBT_PM_NONDOMINANT',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='YBT PM')
ybt_box_plot(data_YBT, x=x,
             y1='YBT_PL_DOMINANT',
             y2='YBT_PL_NONDOMINANT',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='YBT PL')
ybt_box_plot(data_YBT, x=x,
             y1='YBT_COMPOSITE_DOMINANT',
             y2='YBT_COMPOSITE_NONDOMINANT',
             title1=title1,
             title2=title2, 
             xlabel=xlabel,
             ylabel='YBT COMP')

# %%
#FMS
def fms_box_plot(data, x, y, title, xlabel, ylabel):
    data_FMS = data
    bins = pd.cut(data_FMS[x], bins=4, include_lowest=True)
    bin_labels = [f"{round(b.left, 2)}-{round(b.right, 2)}" for b in bins.cat.categories]
    bin_labels[0] = f"{min(data_FMS[x]):.2f}" + bin_labels[0][4:]
    bin_labels = [f"{b}".replace('-', '\u2013') for b in bin_labels]
    for b in bins.cat.categories:
        print(b)
    print(bin_labels)

    data_FMS['Training_Volume_Binned'] = pd.cut(data_FMS[x], bins=4,
                                            labels=bin_labels, include_lowest=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.boxplot(x='Training_Volume_Binned', y=y, data=data_FMS)
    sns.stripplot(x='Training_Volume_Binned', y=y, data=data_FMS, color=".3")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y')
    ax.set_xlabel(xlabel)

    plt.tight_layout()
    plt.show()

title = 'FMS Total Score'
# x = 'Training_Volume_Weekly_MainSport'
# xlabel = 'Training Volume Main Sport [hrs/week]'

x = 'Training_Volume_Weekly_ALLSports'
xlabel = 'Training Volume All Sports [hrs/week]'

ylabel = 'FMS Total Score'

fms_box_plot(data_FMS, x=x, y='FMS_TOTAL', title=title, xlabel=xlabel, ylabel=ylabel)
#%%

# %%