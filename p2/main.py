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
from sklearn.linear_model import LinearRegression, HuberRegressor
import numpy as np
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

# %%
# Plot Venn Diagram
# The regions are identified via three-letter binary codes ('100', '010', etc), hence a valid artgument could look like:
# A tuple with 7 numbers, denoting the sizes of the regions in the following order:
# (100, 010, 110, 001, 101, 011, 111).
lengths = (
    len(data_HHD),
    len(data_YBT),
    len(data_HHD_YBT),
    len(data_FMS),
    len(data_HHD_FMS),
    len(data_YBT_FMS),
    len(data_HHD_YBT_FMS)
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
    data=data_HHD, 
    id_var='Sports_Specialization', 
    value_vars_list=[
        ['PS_DOMINANT_PEAK_FORCE', 'PS_NONDOMINANT_PEAK_FORCE'],
        ['CZ_DOMINANT_PEAK_FORCE', 'CZ_NONDOMINANT_PEAK_FORCE'],
        ['DW_DOMINANT_PEAK_FORCE', 'DW_NONDOMINANT_PEAK_FORCE'],
        ['BR_DOMINANT_PEAK_FORCE', 'BR_NONDOMINANT_PEAK_FORCE']
    ], 
    row_labels=['HHD PS', 'HHD CZ', 'HHD DW', 'HHD BR'], 
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


def cat_box_plot(data, x, y1, y2, title1, title2, xlabel, ylabel, showfliers=False):
    
    # cut dataframe into bins based on x and name the ranges
    median_value = data[x].median()

    bins=[data[x].min(), median_value, data[x].max()]
    print(bins)
    data['Training_Volume_Binned'] = pd.cut(data[x], bins=bins,
                                            include_lowest=True,
                                            labels=['<= Median', '> Median'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    print(data['Training_Volume_Binned'].value_counts())
    # plot boxplot (no outliers cuz stirplot shows doubled points) and stripplot for each y variable
    sns.boxplot(x='Training_Volume_Binned', y=y1, data=data, ax=axes[0], showfliers=showfliers)
    sns.stripplot(x='Training_Volume_Binned', y=y1, data=data, ax=axes[0], color=".3")
    axes[0].set_title(title1)
    axes[0].set_ylabel(ylabel)

    sns.boxplot(x='Training_Volume_Binned', y=y2, data=data, ax=axes[1], showfliers=showfliers)
    sns.stripplot(x='Training_Volume_Binned', y=y2, data=data, ax=axes[1], color=".3")
    axes[1].set_title(title2)
    axes[1].set_ylabel(ylabel)
    ylim_ax0 = axes[0].get_ylim()
    ylim_ax1 = axes[1].get_ylim()
    upper_lim = math.ceil(max(ylim_ax0[1], ylim_ax1[1]))
    lower_lim = math.floor(min(ylim_ax0[0], ylim_ax1[0]))

    if upper_lim <= 10:
        upper_lim = max(ylim_ax0[1], ylim_ax1[1])*1.1

    # Remove individual x-axis labels
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    
    # Set a shared x-axis label for the whole row
    xlabel += f' (median:{median_value})'
    fig.supxlabel(xlabel)

    for ax in axes:
        ax.set_ylim([lower_lim, upper_lim])
        ax.grid(axis='y')

    plt.tight_layout()
    plt.show()

title1 = 'Dominant Extremity'
title2 = 'Non Dominant Extremity'

# x = 'Training_Volume_Weekly_ALLSports'
# xlabel = 'Training Volume All Sports [hrs/week]'

x = 'Training_Volume_Weekly_MainSport'
xlabel = 'Training Volume Main Sport [hrs/week]'

cat_box_plot(data_YBT, x=x,
             y1='YBT_ANT_DOMINANT',
             y2='YBT_ANT_NONDOMINANT',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='YBT ANT')
cat_box_plot(data_YBT, x=x,
             y1='YBT_PM_DOMINANT',
             y2='YBT_PM_NONDOMINANT',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='YBT PM')
cat_box_plot(data_YBT, x=x,
             y1='YBT_PL_DOMINANT',
             y2='YBT_PL_NONDOMINANT',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='YBT PL')
cat_box_plot(data_YBT, x=x,
             y1='YBT_COMPOSITE_DOMINANT',
             y2='YBT_COMPOSITE_NONDOMINANT',
             title1=title1,
             title2=title2, 
             xlabel=xlabel,
             ylabel='YBT COMP')

# %%
#FMS
def fms_box_plot(data, x, y, title, xlabel, ylabel, precision=0):
    median_value = data[x].median()
    bins=[data[x].min(), median_value, data[x].max()]

    data['Training_Volume_Binned'] = pd.cut(data[x], bins=bins,
                                            labels=['<= Median', '> Median'],
                                            include_lowest=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.boxplot(x='Training_Volume_Binned', y=y, data=data)
    sns.stripplot(x='Training_Volume_Binned', y=y, data=data, color=".3")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y')
    xlabel += f' (median:{median_value})'
    fig.supxlabel(xlabel)

    plt.tight_layout()
    plt.show()

title = 'FMS Total Score'
x = 'Training_Volume_Weekly_MainSport'
xlabel = 'Training Volume Main Sport [hrs/week]'
ylabel = 'FMS Total Score'
fms_box_plot(data_FMS, x=x, y='FMS_TOTAL', title=title, xlabel=xlabel, ylabel=ylabel)

x = 'Training_Volume_Weekly_ALLSports'
xlabel = 'Training Volume All Sports [hrs/week]'
fms_box_plot(data_FMS, x=x, y='FMS_TOTAL', title=title, xlabel=xlabel, ylabel=ylabel)




# %%
#make linear plot

def linear_plot(data, x, y1, y2, xlabel, ylabel1, ylabel2, title1, title2):
    # sns.set_theme(style="darkgrid")
    X = data[[x]].values.reshape(-1, 1)
    Y1 = data[y1].values
    Y2 = data[y2].values

    model1 = HuberRegressor()
    model1.fit(X, Y1)
    predictions1 = model1.predict(X)
    coef1 = model1.coef_[0]
    intercept1 = model1.intercept_

    model2 = HuberRegressor()
    model2.fit(X, Y2)
    predictions2 = model2.predict(X)
    coef2 = model2.coef_[0]
    intercept2 = model2.intercept_

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.scatterplot(x=X.flatten(), y=Y1, ax=axes[0], color='blue', label='Data points')
    sns.lineplot(x=X.flatten(), y=predictions1, ax=axes[0], color='red', label=f'Linear fit: y = {coef1:.2f}x + {intercept1:.2f}')
    axes[0].set_ylabel(ylabel1)
    axes[0].set_title(title1)
    axes[0].legend()
    axes[0].grid(True)

    sns.scatterplot(x=X.flatten(), y=Y2, ax=axes[1], color='blue', label='Data points')
    sns.lineplot(x=X.flatten(), y=predictions2, ax=axes[1], color='red', label=f'Linear fit: y = {coef2:.2f}x + {intercept2:.2f}')
    axes[1].set_ylabel(ylabel2)
    axes[1].set_title(title2)
    axes[1].legend()
    axes[1].grid(True)

    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    fig.supxlabel(xlabel)
    ylim_ax0 = axes[0].get_ylim()
    ylim_ax1 = axes[1].get_ylim()
    upper_lim = math.ceil(max(ylim_ax0[1], ylim_ax1[1]))
    lower_lim = math.floor(min(ylim_ax0[0], ylim_ax1[0]))
    for ax in axes:
        ax.set_ylim([lower_lim, upper_lim])
        ax.grid
    
    plt.tight_layout()
    plt.show()

    print(f'Coefficient for {y1}: {coef1:.2f}')
    print(f'Intercept for {y1}: {intercept1:.2f}')
    print(f'Coefficient for {y2}: {coef2:.2f}')
    print(f'Intercept for {y2}: {intercept2:.2f}')

linear_plot(data_YBT, x='Training_Volume_Weekly_ALLSports', y1='YBT_ANT_DOMINANT', y2='YBT_ANT_NONDOMINANT', 
            xlabel='Training Volume All Sports [hrs/week]', ylabel1='YBT ANT', ylabel2='YBT ANT', 
            title1='',
            title2='')
linear_plot(data_YBT, x='Training_Volume_Weekly_ALLSports', y1='YBT_PM_DOMINANT', y2='YBT_PM_NONDOMINANT', 
            xlabel='Training Volume All Sports [hrs/week]', ylabel1='YBT PM', ylabel2='YBT PM', 
            title1='',
            title2='')
linear_plot(data_YBT, x='Training_Volume_Weekly_ALLSports', y1='YBT_PL_DOMINANT', y2='YBT_PL_NONDOMINANT', 
            xlabel='Training Volume All Sports [hrs/week]', ylabel1='YBT PL', ylabel2='YBT PL', 
            title1='',
            title2='')
linear_plot(data_YBT, x='Training_Volume_Weekly_ALLSports', y1='YBT_COMPOSITE_DOMINANT', y2='YBT_COMPOSITE_NONDOMINANT', 
            xlabel='Training Volume All Sports [hrs/week]', ylabel1='YBT COMP', ylabel2='YBT COMP', 
            title1='',
            title2='')
# %%

linear_plot(data_YBT, x='Training_Volume_Weekly_MainSport', y1='YBT_ANT_DOMINANT', y2='YBT_ANT_NONDOMINANT', 
            xlabel='Training Volume Main Sport [hrs/week]', ylabel1='YBT ANT', ylabel2='YBT ANT', 
            title1='',
            title2='')
linear_plot(data_YBT, x='Training_Volume_Weekly_MainSport', y1='YBT_PM_DOMINANT', y2='YBT_PM_NONDOMINANT', 
            xlabel='Training Volume Main Sport [hrs/week]', ylabel1='YBT PM', ylabel2='YBT PM', 
            title1='',
            title2='')
linear_plot(data_YBT, x='Training_Volume_Weekly_MainSport', y1='YBT_PL_DOMINANT', y2='YBT_PL_NONDOMINANT', 
            xlabel='Training Volume Main Sport [hrs/week]', ylabel1='YBT PL', ylabel2='YBT PL', 
            title1='',
            title2='')
linear_plot(data_YBT, x='Training_Volume_Weekly_MainSport', y1='YBT_COMPOSITE_DOMINANT', y2='YBT_COMPOSITE_NONDOMINANT', 
            xlabel='Training Volume Main Sport [hrs/week]', ylabel1='YBT COMP', ylabel2='YBT COMP', 
            title1='',
            title2='')


# %%
def linear_plot_fms(data, x, y, xlabel, ylabel, title):
    # sns.set_theme(style="darkgrid")
    X = data[[x]].values.reshape(-1, 1)
    Y = data[y].values

    model = HuberRegressor()
    model.fit(X, Y)
    predictions = model.predict(X)
    coef = model.coef_[0]
    intercept = model.intercept_

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X.flatten(), y=Y, color='blue', label='Data points')
    sns.lineplot(x=X.flatten(), y=predictions, color='red', label=f'Linear fit: y = {coef:.2f}x + {intercept:.2f}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f'Coefficient: {coef:.2f}')
    print(f'Intercept: {intercept:.2f}')

linear_plot_fms(data_FMS, x='Training_Volume_Weekly_ALLSports', y='FMS_TOTAL', 
                xlabel='Training Volume All Sports [hrs/week]', ylabel='FMS Total Score', 
                title='')
linear_plot_fms(data_FMS, x='Training_Volume_Weekly_MainSport', y='FMS_TOTAL', 
                xlabel='Training Volume Main Sport [hrs/week]', ylabel='FMS Total Score', 
                title='')
#%%
# Hypothesis 3
#H3 – Zawodnicy trenujący więcej niż mają lat (Athletes who participated in their primary sport
# for more hours per week than their age (Yes/No) = Yes) będą wykazywać niższe wyniki w
# Sports Performance Tests dla dominującej i niedominującej kończyny dolnej (YBT, HHD, FMS)
def plot_dominant_vs_nondominant(data, x, y_vars, ylabels, xlabel, hue_label):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        df_melted = data.melt(id_vars=[x],
                              value_vars=[y_vars[2*i], y_vars[2*i+1]],
                              var_name=hue_label, value_name='Score')

        sns.boxplot(x=x, y='Score', hue=hue_label, data=df_melted, ax=ax, showfliers=False,
                    gap=0.5, order=['No', 'Yes'])
        sns.stripplot(x=x, y='Score', hue=hue_label, data=df_melted, ax=ax,
                      color=".3", dodge=True)
        ax.set_ylabel(ylabels[i])
        ax.set_xlabel('')
        ax.grid(axis='y')
        ax.legend().remove()

    handles, _ = axes[0, 0].get_legend_handles_labels()  # Get handles and labels from one of the plots
    fig.legend(handles, ['Dominant', 'Nondominant'], loc='lower center', title=hue_label,
               ncol=2, bbox_to_anchor=(0.5, 1), frameon=False)
    fig.supxlabel(xlabel)
    plt.tight_layout()
    plt.show()

y_vars = [
    'YBT_ANT_DOMINANT', 'YBT_ANT_NONDOMINANT',
    'YBT_PM_DOMINANT', 'YBT_PM_NONDOMINANT',
    'YBT_PL_DOMINANT', 'YBT_PL_NONDOMINANT',
    'YBT_COMPOSITE_DOMINANT', 'YBT_COMPOSITE_NONDOMINANT'
]
xlabel = 'Athletes who participated in their primary sport for more hours per week than their age (Yes/No)'
ylabels = ['YBT ANT', 'YBT PM', 'YBT PL', 'YBT COMP']
plot_dominant_vs_nondominant(data_YBT, x='Hours_per_week>Age',
                             y_vars=y_vars, ylabels=ylabels,
                             xlabel=xlabel,
                             hue_label='Dominant Extremity')
# %%
y_vars = ['PS_DOMINANT_PEAK_FORCE', 'PS_NONDOMINANT_PEAK_FORCE',
            'CZ_DOMINANT_PEAK_FORCE', 'CZ_NONDOMINANT_PEAK_FORCE',
            'DW_DOMINANT_PEAK_FORCE', 'DW_NONDOMINANT_PEAK_FORCE',
            'BR_DOMINANT_PEAK_FORCE', 'BR_NONDOMINANT_PEAK_FORCE']

ylabels = ['HHD PS', 'HHD CZ', 'HHD DW', 'HHD BR']
plot_dominant_vs_nondominant(data_HHD, x='Hours_per_week>Age',
                             y_vars=y_vars, ylabels=ylabels,
                             xlabel=xlabel,
                             hue_label='Dominant Extremity')
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
sns.boxplot(x='Hours_per_week>Age', y='FMS_TOTAL', data=data_FMS, ax=ax)
sns.stripplot(x='Hours_per_week>Age', y='FMS_TOTAL', data=data_FMS, color=".3", ax=ax)
plt.xlabel(xlabel)
plt.ylabel('FMS Total Score')
plt.tight_layout()
ax.grid(axis='y')
plt.show()
# %%
# Hypothesis 4

# H4 - Zawodnicy z historią urazów mają niższe wyniki w Sports Performance Tests dla
# dominującej i niedominującej kończyny dolnej (YBT, HHD, FMS)

y_vars = ['YBT_ANT_DOMINANT', 'YBT_ANT_NONDOMINANT',
            'YBT_PM_DOMINANT', 'YBT_PM_NONDOMINANT',
            'YBT_PL_DOMINANT', 'YBT_PL_NONDOMINANT',
            'YBT_COMPOSITE_DOMINANT', 'YBT_COMPOSITE_NONDOMINANT']

ylabels = ['YBT ANT', 'YBT PM', 'YBT PL', 'YBT COMP']

plot_dominant_vs_nondominant(data_YBT, x='Injury_History',
                             y_vars=y_vars, ylabels=ylabels,
                             xlabel='Injury History (Yes/No)',
                             hue_label='Dominant Extremity')


plot_dominant_vs_nondominant(data_YBT, x='Injury_History_MoreThanOne (0=no,1=yes)',
                                y_vars=y_vars, ylabels=ylabels,
                                xlabel='Injury History More Than One (Yes/No)',
                                hue_label='Dominant Extremity')

y_vars = ['PS_DOMINANT_PEAK_FORCE', 'PS_NONDOMINANT_PEAK_FORCE',
            'CZ_DOMINANT_PEAK_FORCE', 'CZ_NONDOMINANT_PEAK_FORCE',
            'DW_DOMINANT_PEAK_FORCE', 'DW_NONDOMINANT_PEAK_FORCE',
            'BR_DOMINANT_PEAK_FORCE', 'BR_NONDOMINANT_PEAK_FORCE']

ylabels = ['HHD PS', 'HHD CZ', 'HHD DW', 'HHD BR']
plot_dominant_vs_nondominant(data_HHD, x='Injury_History',
                             y_vars=y_vars, ylabels=ylabels,
                             xlabel='Injury History (Yes/No)',
                             hue_label='Dominant Extremity')

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
sns.boxplot(x='Injury_History', y='FMS_TOTAL', data=data_FMS, ax=ax, order=['No', 'Yes'])
sns.stripplot(x='Injury_History', y='FMS_TOTAL', data=data_FMS, color=".3", ax=ax)
plt.xlabel('Injury History (Yes/No)')
plt.ylabel('FMS Total Score')
plt.tight_layout()
ax.grid(axis='y')
plt.show()

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
sns.boxplot(x='Injury_History_MoreThanOne (0=no,1=yes)', y='FMS_TOTAL', data=data_FMS, ax=ax, order=['No', 'Yes'])
sns.stripplot(x='Injury_History_MoreThanOne (0=no,1=yes)', y='FMS_TOTAL', data=data_FMS, color=".3", ax=ax)
plt.xlabel('Injury History More Than One (Yes/No)')
plt.ylabel('FMS Total Score')
plt.tight_layout()
ax.grid(axis='y')
plt.show()
#%%
#Hypothesis 6 (ommit 5 - no data)

x = 'Chronologic_Age'
xlabel = 'Chronologic Age [years]'


cat_box_plot(data_YBT, x=x,
             y1='YBT_ANT_DOMINANT',
             y2='YBT_ANT_NONDOMINANT',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='YBT ANT')
cat_box_plot(data_YBT, x=x,
             y1='YBT_PM_DOMINANT',
             y2='YBT_PM_NONDOMINANT',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='YBT PM')
cat_box_plot(data_YBT, x=x,
             y1='YBT_PL_DOMINANT',
             y2='YBT_PL_NONDOMINANT',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='YBT PL')
cat_box_plot(data_YBT, x=x,
             y1='YBT_COMPOSITE_DOMINANT',
             y2='YBT_COMPOSITE_NONDOMINANT',
             title1=title1,
             title2=title2, 
             xlabel=xlabel,
             ylabel='YBT COMP')
# %%
cat_box_plot(data_HHD, x=x,
             y1='PS_DOMINANT_PEAK_FORCE',
             y2='PS_NONDOMINANT_PEAK_FORCE',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='HHD PS')
cat_box_plot(data_HHD, x=x,
             y1='CZ_DOMINANT_PEAK_FORCE',
             y2='CZ_NONDOMINANT_PEAK_FORCE',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='HHD CZ')
cat_box_plot(data_HHD, x=x,
             y1='DW_DOMINANT_PEAK_FORCE',
             y2='DW_NONDOMINANT_PEAK_FORCE',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='HHD DW')
cat_box_plot(data_HHD, x=x,
             y1='BR_DOMINANT_PEAK_FORCE',
             y2='BR_NONDOMINANT_PEAK_FORCE',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='HHD BR')

#FMS
fms_box_plot(data_FMS, x=x, y='FMS_TOTAL', title=title, xlabel=xlabel, ylabel=ylabel)

# %%
# # 4 sport specialization criteria:
# 'Given_up_sport_for_main', 'Main_sport_more_important',
# 'Months_in_a_year>8', 'Hours_per_week>Age',

create_boxplot_4row(
    data=data_YBT, 
    id_var='Given_up_sport_for_main', 
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
    order=['No', 'Yes'], 
    x_label='Given up sport for main sport'
)
create_boxplot_4row(
    data=data_YBT, 
    id_var='Main_sport_more_important', 
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
    order=['No', 'Yes'], 
    x_label='Main sport more important'
)
create_boxplot_4row(
    data=data_YBT, 
    id_var='Months_in_a_year>8', 
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
    order=['No', 'Yes'], 
    x_label='Trains for more than 8 months in a year'
)
create_boxplot_4row(
    data=data_YBT, 
    id_var='Hours_per_week>Age', 
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
    order=['No', 'Yes'], 
    x_label='Athletes who participated in their primary sport for more hours per week than their age'
)