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
# PL => ENG acronyms
#  PŚ => HAbd
#  CZ => KE
#  DW => KF
#  BR => AF
HHD = [
    'HAbd_L_PEAK_FORCE',
    'HAbd_R_PEAK_FORCE',
    'KE_L_PEAK_FORCE',
    'KE_R_PEAK_FORCE',
    'KF_L_PEAK_FORCE',
    'KF_R_PEAK_FORCE',
    'AF_L_PEAK_FORCE',
    'AF_R_PEAK_FORCE'
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


data_HHD['HAbd_DOMINANT_PEAK_FORCE'] = data_HHD.apply(assign_dominant_extremity, left_col='HAbd_L_PEAK_FORCE', right_col='HAbd_R_PEAK_FORCE', axis=1)
data_HHD['KE_DOMINANT_PEAK_FORCE'] = data_HHD.apply(assign_dominant_extremity, left_col='KE_L_PEAK_FORCE', right_col='KE_R_PEAK_FORCE', axis=1)
data_HHD['KF_DOMINANT_PEAK_FORCE'] = data_HHD.apply(assign_dominant_extremity, left_col='KF_L_PEAK_FORCE', right_col='KF_R_PEAK_FORCE', axis=1)
data_HHD['AF_DOMINANT_PEAK_FORCE'] = data_HHD.apply(assign_dominant_extremity, left_col='AF_L_PEAK_FORCE', right_col='AF_R_PEAK_FORCE', axis=1)

data_HHD['HAbd_NONDOMINANT_PEAK_FORCE'] = data_HHD.apply(assign_dominant_extremity, right_col='HAbd_L_PEAK_FORCE', left_col='HAbd_R_PEAK_FORCE', axis=1)
data_HHD['KE_NONDOMINANT_PEAK_FORCE'] = data_HHD.apply(assign_dominant_extremity, right_col='KE_L_PEAK_FORCE', left_col='KE_R_PEAK_FORCE', axis=1)
data_HHD['KF_NONDOMINANT_PEAK_FORCE'] = data_HHD.apply(assign_dominant_extremity, right_col='KF_L_PEAK_FORCE', left_col='KF_R_PEAK_FORCE', axis=1)
data_HHD['AF_NONDOMINANT_PEAK_FORCE'] = data_HHD.apply(assign_dominant_extremity, right_col='AF_L_PEAK_FORCE', left_col='AF_R_PEAK_FORCE', axis=1)


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
    row_labels=['YBT ANT [%]', 'YBT PM [%]', 'YBT PL [%]', 'YBT COMP [%]'], 
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
        ['HAbd_DOMINANT_PEAK_FORCE', 'HAbd_NONDOMINANT_PEAK_FORCE'],
        ['KE_DOMINANT_PEAK_FORCE', 'KE_NONDOMINANT_PEAK_FORCE'],
        ['KF_DOMINANT_PEAK_FORCE', 'KF_NONDOMINANT_PEAK_FORCE'],
        ['AF_DOMINANT_PEAK_FORCE', 'AF_NONDOMINANT_PEAK_FORCE']
    ], 
    row_labels=['HHD HAbd [kg]', 'HHD KE [kg]', 'HHD KF [kg]', 'HHD AF [kg]'], 
    hue_label='Dominant Extremity', 
    legend_labels=['Dominant', 'Nondominant'], 
    palette=["m", "g"], 
    order=['low', 'moderate', 'high'], 
    x_label='Sports Specialization'
)

# %%
fig, ax = plt.subplots(figsize=(5, 4))
sns.boxplot(x='Sports_Specialization', y='FMS_TOTAL', data=data_FMS, ax=ax,
            order=['low', 'moderate', 'high'], showfliers=False,
            fill=False, linewidth=1, color='black', width=0.5,
            medianprops=dict(color='gray', linewidth=2))
sns.stripplot(x='Sports_Specialization', y='FMS_TOTAL', data=data_FMS, ax=ax,
              color="black", size=3)
plt.xlabel('Sports Specialization')
plt.ylabel('FMS Total Score [-]')
plt.grid(axis='y')
sns.despine(bottom = True, left = True)
ax.tick_params(left=False, bottom=False)
plt.tight_layout()
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
                                            # labels=['<= Median', '> Median'])
                                            # labels=['$\leq$median', '$>$median']
                                            labels=[f'$\leq$'+str(median_value),
                                                    '$>$'+str(median_value)],)

    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    print(data['Training_Volume_Binned'].value_counts())
    # plot boxplot (no outliers cuz stirplot shows doubled points) and stripplot for each y variable
    sns.boxplot(x='Training_Volume_Binned', y=y1, data=data, ax=axes[0], showfliers=showfliers,
                fill=False, linewidth=1, color='black', width=0.5,
                medianprops=dict(color='gray', linewidth=2))
    sns.stripplot(x='Training_Volume_Binned', y=y1, data=data, ax=axes[0],
                  color="black", size=3)
    axes[0].set_title(title1)
    axes[0].set_ylabel(ylabel)

    sns.boxplot(x='Training_Volume_Binned', y=y2, data=data, ax=axes[1], showfliers=showfliers,
                fill=False, linewidth=1, color='black', width=0.5,
                medianprops=dict(color='gray', linewidth=2))
    sns.stripplot(x='Training_Volume_Binned', y=y2, data=data, ax=axes[1],
                  color="black", size=3)
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
    axes[1].set_ylabel('')
    # Set a shared x-axis label for the whole row
    fig.supxlabel(xlabel)

    for ax in axes:
        ax.set_ylim([lower_lim, upper_lim])
        ax.grid(axis='y')

    sns.despine(bottom = True, left = True)
    for ax in axes:
        ax.tick_params(left=False, bottom=False)
    
    axes[1].tick_params(axis='y', which='both', left=False, labelleft=False)

    plt.tight_layout()
    plt.show()

title1 = 'Dominant Extremity'
title2 = 'Non Dominant Extremity'

x = 'Training_Volume_Weekly_ALLSports'
xlabel = 'Training Volume All Sports [hrs/week]'

x = 'Training_Volume_Weekly_MainSport'
xlabel = 'Training Volume Main Sport [hrs/week]'

cat_box_plot(data_YBT, x=x,
             y1='YBT_ANT_DOMINANT',
             y2='YBT_ANT_NONDOMINANT',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='YBT ANT [%]')
cat_box_plot(data_YBT, x=x,
             y1='YBT_PM_DOMINANT',
             y2='YBT_PM_NONDOMINANT',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='YBT PM [%]')
cat_box_plot(data_YBT, x=x,
             y1='YBT_PL_DOMINANT',
             y2='YBT_PL_NONDOMINANT',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='YBT PL [%]')
cat_box_plot(data_YBT, x=x,
             y1='YBT_COMPOSITE_DOMINANT',
             y2='YBT_COMPOSITE_NONDOMINANT',
             title1=title1,
             title2=title2, 
             xlabel=xlabel,
             ylabel='YBT COMP [%]')

# %%
cat_box_plot(data_HHD, x=x,
                y1='HAbd_DOMINANT_PEAK_FORCE',
                y2='HAbd_NONDOMINANT_PEAK_FORCE',
                title1=title1,
                title2=title2,
                xlabel=xlabel,
                ylabel='HHD HAbd [kg]')
cat_box_plot(data_HHD, x=x,
                y1='KE_DOMINANT_PEAK_FORCE',
                y2='KE_NONDOMINANT_PEAK_FORCE',
                title1=title1,
                title2=title2,
                xlabel=xlabel,
                ylabel='HHD KE [kg]')
cat_box_plot(data_HHD, x=x,
                y1='KF_DOMINANT_PEAK_FORCE',
                y2='KF_NONDOMINANT_PEAK_FORCE',
                title1=title1,
                title2=title2,
                xlabel=xlabel,
                ylabel='HHD KF [kg]')
cat_box_plot(data_HHD, x=x,
                y1='AF_DOMINANT_PEAK_FORCE',
                y2='AF_NONDOMINANT_PEAK_FORCE',
                title1=title1,
                title2=title2,
                xlabel=xlabel,
                ylabel='HHD AF [kg]')

# %%
#FMS
def fms_box_plot(data, x, y, title, xlabel, ylabel, precision=0):
    median_value = data[x].median()
    bins=[data[x].min(), median_value, data[x].max()]

    data['Training_Volume_Binned'] = pd.cut(data[x], bins=bins,
                                            labels=[f'$\leq$'+str(median_value),
                                                    '$>$'+str(median_value)],
                                            include_lowest=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(x='Training_Volume_Binned', y=y, data=data, showfliers=False,
                fill=False, linewidth=1, color='black', width=0.25,
                medianprops=dict(color='gray', linewidth=2))
    sns.stripplot(x='Training_Volume_Binned', y=y, data=data,
                  color="black", size=3)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(axis='y')

    sns.despine(bottom = True, left = True)
    ax.tick_params(left=False, bottom=False)
    plt.tight_layout()
    plt.show()

title = 'Comparison of FMS Total Scores by Weekly Training Volume'
x = 'Training_Volume_Weekly_MainSport'
xlabel = 'Training Volume Main Sport [hrs/week]'
ylabel = 'FMS Total Score [-]'
fms_box_plot(data_FMS, x=x, y='FMS_TOTAL', title=title, xlabel=xlabel, ylabel=ylabel)

x = 'Training_Volume_Weekly_ALLSports'
xlabel = 'Training Volume All Sports'
fms_box_plot(data_FMS, x=x, y='FMS_TOTAL', title=title, xlabel=xlabel, ylabel=ylabel)


# %%
#make linear plot

def linear_plot(data, x, y1, y2, xlabel, ylabel1, ylabel2, title1, title2):
    # sns.set_theme(style="darkgrid")
    X = data[[x]].values.reshape(-1, 1)
    Y1 = data[y1].values
    Y2 = data[y2].values

    # model1 = HuberRegressor()
    model1 = LinearRegression()
    model1.fit(X, Y1)
    predictions1 = model1.predict(X)
    coef1 = model1.coef_[0]
    intercept1 = model1.intercept_

    model2 = HuberRegressor()
    model2.fit(X, Y2)
    predictions2 = model2.predict(X)
    coef2 = model2.coef_[0]
    intercept2 = model2.intercept_

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    sns.lineplot(x=X.flatten(), y=predictions1, ax=axes[0], color='black', linewidth=4,
                 label=f'Linear fit: y = {coef1:.2f}x + {intercept1:.2f}')
    axes[0].legend()
    sns.scatterplot(x=X.flatten(), y=Y1, ax=axes[0], color='gray')
    axes[0].set_ylabel(ylabel1)
    axes[0].set_xlabel(xlabel)
    axes[0].set_title(title1)
    axes[0].grid(True)

    sns.lineplot(x=X.flatten(), y=predictions2, ax=axes[1], color='black', linewidth=4,
                 label=f'Linear fit: y = {coef2:.2f}x + {intercept2:.2f}')
    axes[1].legend()
    sns.scatterplot(x=X.flatten(), y=Y2, ax=axes[1], color='gray')
    axes[1].set_ylabel(ylabel2)
    axes[1].set_xlabel(xlabel)
    axes[1].set_title(title2)
    axes[1].grid(True)

    axes[0].set_title('Dominant Extremity')
    axes[1].set_title('Non Dominant Extremity')
    # fig.supxlabel(xlabel)
    ylim_ax0 = axes[0].get_ylim()
    ylim_ax1 = axes[1].get_ylim()
    upper_lim = math.ceil(max(ylim_ax0[1], ylim_ax1[1]))
    lower_lim = math.floor(min(ylim_ax0[0], ylim_ax1[0]))
    for ax in axes:
        ax.set_ylim([lower_lim, upper_lim])
        ax.grid

    sns.despine(bottom = True, left = True)
    ax.tick_params(left=False, bottom=False)
    plt.tight_layout()
    plt.show()

    print(f'Coefficient for {y1}: {coef1:.2f}')
    print(f'Intercept for {y1}: {intercept1:.2f}')
    print(f'Coefficient for {y2}: {coef2:.2f}')
    print(f'Intercept for {y2}: {intercept2:.2f}')

linear_plot(data_YBT, x='Training_Volume_Weekly_ALLSports', y1='YBT_ANT_DOMINANT', y2='YBT_ANT_NONDOMINANT', 
            xlabel='Training Volume All Sports [hrs/week]', ylabel1='YBT ANT [%]', ylabel2='YBT ANT [%]', 
            title1='',
            title2='')
linear_plot(data_YBT, x='Training_Volume_Weekly_ALLSports', y1='YBT_PM_DOMINANT', y2='YBT_PM_NONDOMINANT', 
            xlabel='Training Volume All Sports [hrs/week]', ylabel1='YBT PM [%]', ylabel2='YBT PM [%]', 
            title1='',
            title2='')
linear_plot(data_YBT, x='Training_Volume_Weekly_ALLSports', y1='YBT_PL_DOMINANT', y2='YBT_PL_NONDOMINANT', 
            xlabel='Training Volume All Sports [hrs/week]', ylabel1='YBT PL [%]', ylabel2='YBT PL [%]', 
            title1='',
            title2='')
linear_plot(data_YBT, x='Training_Volume_Weekly_ALLSports', y1='YBT_COMPOSITE_DOMINANT', y2='YBT_COMPOSITE_NONDOMINANT', 
            xlabel='Training Volume All Sports [hrs/week]', ylabel1='YBT COMP [%]', ylabel2='YBT COMP [%]', 
            title1='',
            title2='')
# %%

linear_plot(data_YBT, x='Training_Volume_Weekly_MainSport', y1='YBT_ANT_DOMINANT', y2='YBT_ANT_NONDOMINANT', 
            xlabel='Training Volume Main Sport [hrs/week]', ylabel1='YBT ANT [%]', ylabel2='YBT ANT [%]', 
            title1='',
            title2='')
linear_plot(data_YBT, x='Training_Volume_Weekly_MainSport', y1='YBT_PM_DOMINANT', y2='YBT_PM_NONDOMINANT', 
            xlabel='Training Volume Main Sport [hrs/week]', ylabel1='YBT PM [%]', ylabel2='YBT PM [%]', 
            title1='',
            title2='')
linear_plot(data_YBT, x='Training_Volume_Weekly_MainSport', y1='YBT_PL_DOMINANT', y2='YBT_PL_NONDOMINANT', 
            xlabel='Training Volume Main Sport [hrs/week]', ylabel1='YBT PL [%]', ylabel2='YBT PL [%]', 
            title1='',
            title2='')
linear_plot(data_YBT, x='Training_Volume_Weekly_MainSport', y1='YBT_COMPOSITE_DOMINANT', y2='YBT_COMPOSITE_NONDOMINANT', 
            xlabel='Training Volume Main Sport [hrs/week]', ylabel1='YBT COMP [%]', ylabel2='YBT COMP [%]', 
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

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.lineplot(x=X.flatten(), y=predictions, color='black', linewidth=4,
                 label=f'Linear fit: y = {coef:.2f}x + {intercept:.2f}', ax=ax)
    sns.scatterplot(x=X.flatten(), y=Y, color='gray')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    sns.despine(bottom = True, left = True)
    ax.tick_params(left=False, bottom=False)
    plt.tight_layout()
    plt.show()

    print(f'Coefficient: {coef:.2f}')
    print(f'Intercept: {intercept:.2f}')

linear_plot_fms(data_FMS, x='Training_Volume_Weekly_ALLSports', y='FMS_TOTAL', 
                xlabel='Training Volume All Sports [hrs/week]', ylabel='FMS Total Score [-]', 
                title='')
linear_plot_fms(data_FMS, x='Training_Volume_Weekly_MainSport', y='FMS_TOTAL', 
                xlabel='Training Volume Main Sport [hrs/week]', ylabel='FMS Total Score [-]', 
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
xlabel = 'Athletes who participated in their primary sport for more hrs/week than their age (Yes/No)'
ylabels = ['YBT ANT [%]', 'YBT PM [%]', 'YBT PL [%]', 'YBT COMP [%]']
plot_dominant_vs_nondominant(data_YBT, x='Hours_per_week>Age',
                             y_vars=y_vars, ylabels=ylabels,
                             xlabel=xlabel,
                             hue_label='Dominant Extremity')
# %%
y_vars =['HAbd_DOMINANT_PEAK_FORCE', 'HAbd_NONDOMINANT_PEAK_FORCE',
        'KE_DOMINANT_PEAK_FORCE', 'KE_NONDOMINANT_PEAK_FORCE',
        'KF_DOMINANT_PEAK_FORCE', 'KF_NONDOMINANT_PEAK_FORCE',
        'AF_DOMINANT_PEAK_FORCE', 'AF_NONDOMINANT_PEAK_FORCE']

ylabels = ['HHD HAbd [kg]', 'HHD KE [kg]', 'HHD KF [kg]', 'HHD AF [kg]']
plot_dominant_vs_nondominant(data_HHD, x='Hours_per_week>Age',
                             y_vars=y_vars, ylabels=ylabels,
                             xlabel=xlabel,
                             hue_label='Dominant Extremity')
# %%

fig, ax = plt.subplots(figsize=(5, 4))
sns.boxplot(x='Hours_per_week>Age', y='FMS_TOTAL', data=data_FMS, ax=ax,
            showfliers=False,  order=['No', 'Yes'],
            fill=False, linewidth=1, color='black', width=0.5,
            medianprops=dict(color='gray', linewidth=2))
sns.stripplot(x='Hours_per_week>Age', y='FMS_TOTAL', data=data_FMS, ax=ax,
              color="black", size=3)
xlabel = 'Athletes who participated in their primary sport\nfor more hrs/week than their age (Yes/No)'
plt.xlabel(xlabel)
plt.ylabel('FMS Total Score [-]')
plt.grid(axis='y')
sns.despine(bottom = True, left = True)
ax.tick_params(left=False, bottom=False)
plt.tight_layout()
plt.show()

# %%
# Hypothesis 4

# H4 - Zawodnicy z historią urazów mają niższe wyniki w Sports Performance Tests dla
# dominującej i niedominującej kończyny dolnej (YBT, HHD, FMS)

y_vars = ['YBT_ANT_DOMINANT', 'YBT_ANT_NONDOMINANT',
          'YBT_PM_DOMINANT', 'YBT_PM_NONDOMINANT',
          'YBT_PL_DOMINANT', 'YBT_PL_NONDOMINANT',
          'YBT_COMPOSITE_DOMINANT', 'YBT_COMPOSITE_NONDOMINANT']

ylabels = ['YBT ANT [%]', 'YBT PM [%]', 'YBT PL [%]', 'YBT COMP [%]']

plot_dominant_vs_nondominant(data_YBT, x='Injury_History',
                             y_vars=y_vars, ylabels=ylabels,
                             xlabel='Injury History (Yes/No)',
                             hue_label='Dominant Extremity')


plot_dominant_vs_nondominant(data_YBT, x='Injury_History_MoreThanOne (0=no,1=yes)',
                                y_vars=y_vars, ylabels=ylabels,
                                xlabel='Injury History More Than One (Yes/No)',
                                hue_label='Dominant Extremity')

y_vars =['HAbd_DOMINANT_PEAK_FORCE', 'HAbd_NONDOMINANT_PEAK_FORCE',
         'KE_DOMINANT_PEAK_FORCE', 'KE_NONDOMINANT_PEAK_FORCE',
         'KF_DOMINANT_PEAK_FORCE', 'KF_NONDOMINANT_PEAK_FORCE',
         'AF_DOMINANT_PEAK_FORCE', 'AF_NONDOMINANT_PEAK_FORCE']

ylabels = ['HHD HAbd [kg]', 'HHD KE [kg]', 'HHD KF [kg]', 'HHD AF [kg]']
plot_dominant_vs_nondominant(data_HHD, x='Injury_History',
                             y_vars=y_vars, ylabels=ylabels,
                             xlabel='Injury History (Yes/No)',
                             hue_label='Dominant Extremity')
plot_dominant_vs_nondominant(data_HHD, x='Injury_History_MoreThanOne (0=no,1=yes)',
                             y_vars=y_vars, ylabels=ylabels,
                             xlabel='Injury History More Than One (0=no,1=yes)',
                             hue_label='Dominant Extremity')
# %%
fig, ax = plt.subplots(figsize=(5, 4))
sns.boxplot(x='Injury_History', y='FMS_TOTAL', data=data_FMS, ax=ax,
            showfliers=False, order=['No', 'Yes'],
            fill=False, linewidth=1, color='black', width=0.5,
            medianprops=dict(color='gray', linewidth=2))
sns.stripplot(x='Injury_History', y='FMS_TOTAL', data=data_FMS, ax=ax,
              color="black", size=3)
plt.xlabel('Injury History (Yes/No)')
plt.ylabel('FMS Total Score [-]')
plt.grid(axis='y')
sns.despine(bottom = True, left = True)
ax.tick_params(left=False, bottom=False)
plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(5, 4))
sns.boxplot(x='Injury_History_MoreThanOne (0=no,1=yes)', y='FMS_TOTAL', data=data_FMS, ax=ax,
            showfliers=False, order=['No', 'Yes'],
            fill=False, linewidth=1, color='black', width=0.5,
            medianprops=dict(color='gray', linewidth=2))
sns.stripplot(x='Injury_History_MoreThanOne (0=no,1=yes)', y='FMS_TOTAL', data=data_FMS, ax=ax,
              color="black", size=3)
plt.xlabel('Injury History More Than One (Yes/No)')
plt.ylabel('FMS Total Score [-]')
plt.grid(axis='y')
sns.despine(bottom = True, left = True)
ax.tick_params(left=False, bottom=False)
plt.tight_layout()
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
             ylabel='YBT ANT [%]')
cat_box_plot(data_YBT, x=x,
             y1='YBT_PM_DOMINANT',
             y2='YBT_PM_NONDOMINANT',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='YBT PM [%]')
cat_box_plot(data_YBT, x=x,
             y1='YBT_PL_DOMINANT',
             y2='YBT_PL_NONDOMINANT',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='YBT PL [%]')
cat_box_plot(data_YBT, x=x,
             y1='YBT_COMPOSITE_DOMINANT',
             y2='YBT_COMPOSITE_NONDOMINANT',
             title1=title1,
             title2=title2, 
             xlabel=xlabel,
             ylabel='YBT COMP [%]')
# %%
cat_box_plot(data_HHD, x=x,
             y1='HAbd_DOMINANT_PEAK_FORCE',
             y2='HAbd_NONDOMINANT_PEAK_FORCE',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='HHD HAbd [kg]')
cat_box_plot(data_HHD, x=x,
             y1='KE_DOMINANT_PEAK_FORCE',
             y2='KE_NONDOMINANT_PEAK_FORCE',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='HHD KE [kg]')
cat_box_plot(data_HHD, x=x,
             y1='KF_DOMINANT_PEAK_FORCE',
             y2='KF_NONDOMINANT_PEAK_FORCE',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='HHD KF [kg]')
cat_box_plot(data_HHD, x=x,
             y1='AF_DOMINANT_PEAK_FORCE',
             y2='AF_NONDOMINANT_PEAK_FORCE',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='HHD AF [kg]')

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
    row_labels=['YBT ANT [%]', 'YBT PM [%]', 'YBT PL [%]', 'YBT COMP [%]'], 
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
    row_labels=['YBT ANT [%]', 'YBT PM [%]', 'YBT PL [%]', 'YBT COMP [%]'], 
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
    row_labels=['YBT ANT [%]', 'YBT PM [%]', 'YBT PL [%]', 'YBT COMP [%]'], 
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
    row_labels=['YBT ANT [%]', 'YBT PM [%]', 'YBT PL [%]', 'YBT COMP [%]'], 
    hue_label='Dominant Extremity', 
    legend_labels=['Dominant', 'Nondominant'], 
    palette=["m", "g"], 
    order=['No', 'Yes'], 
    x_label='Athletes who participated in their primary sport for more hours per week than their age'
)

# %%

title1 = 'Dominant Extremity'
title2 = 'Non Dominant Extremity'

x = 'Experience_main_sport'
xlabel = 'Experience in main sport [years]'
cat_box_plot(data_YBT, x=x,
             y1='YBT_ANT_DOMINANT',
             y2='YBT_ANT_NONDOMINANT',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='YBT ANT [%]')
cat_box_plot(data_YBT, x=x,
             y1='YBT_PM_DOMINANT',
             y2='YBT_PM_NONDOMINANT',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='YBT PM [%]')
cat_box_plot(data_YBT, x=x,
             y1='YBT_PL_DOMINANT',
             y2='YBT_PL_NONDOMINANT',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='YBT PL [%]')
cat_box_plot(data_YBT, x=x,
             y1='YBT_COMPOSITE_DOMINANT',
             y2='YBT_COMPOSITE_NONDOMINANT',
             title1=title1,
             title2=title2, 
             xlabel=xlabel,
             ylabel='YBT COMP [%]')

# %%

cat_box_plot(data_HHD, x=x,
             y1='HAbd_DOMINANT_PEAK_FORCE',
             y2='HAbd_NONDOMINANT_PEAK_FORCE',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='HHD HAbd [kg]')
cat_box_plot(data_HHD, x=x,
             y1='KE_DOMINANT_PEAK_FORCE',
             y2='KE_NONDOMINANT_PEAK_FORCE',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='HHD KE [kg]')
cat_box_plot(data_HHD, x=x,
             y1='KF_DOMINANT_PEAK_FORCE',
             y2='KF_NONDOMINANT_PEAK_FORCE',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='HHD KF [kg]')
cat_box_plot(data_HHD, x=x,
             y1='AF_DOMINANT_PEAK_FORCE',
             y2='AF_NONDOMINANT_PEAK_FORCE',
             title1=title1,
             title2=title2,
             xlabel=xlabel,
             ylabel='HHD AF [kg]')
# %%
fms_box_plot(data_FMS, x=x, y='FMS_TOTAL', title=title, xlabel=xlabel, ylabel=ylabel)

# %%
# 'Sex', 'Chronologic_Age', 'Sport', 'Sports', 'Dominant_extremity',
#        'Geographic_Factor', 'Pain_now', 'Injury_History',
#        'Injury_History_MoreThanOne (0=no,1=yes)',
#        'Training_Volume_Weekly_MainSport', 'Training_Volume_Weekly_ALLSports',
#        'Given_up_sport_for_main', 'Main_sport_more_important',
#        'Months_in_a_year>8', 'Hours_per_week>Age', 'Experience_main_sport',
#        'Sports_Specialization', 'Sports_Specialization_ordinal',

# d = data_pure
# d = data_YBT
# d = data_HHD
d = data_FMS

key = 'Sex'
key = 'Geographic_Factor'
key = 'Sports_Specialization'
key = 'Dominant_extremity'
key = 'Injury_History'
key = 'Injury_History_MoreThanOne (0=no,1=yes)'

print(d[key].value_counts())
print(f"{round(d[key].value_counts(normalize=True),2)}")
# %%

print(f"{d['Chronologic_Age'].mean():.2f}", end=' (')
print(f"{d['Chronologic_Age'].std():.2f}", end=')\n')


# %%
# plot dominant vs non dominant for all measures
def draw_box_plots(data, pairs, ylabels, figsize=(12, 6), save_path=None):
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    for i, (dominant, non_dominant) in enumerate(pairs):
        melted_data = data[[dominant, non_dominant]].melt(var_name='Group', value_name='Measure')
        # mapt Dominant and Nondominant to group names depending on the name of group
        melted_data['Group'] = melted_data['Group'].apply(lambda x: 'Nondominant' if 'NONDOMINANT' in x else 'Dominant')
        sns.boxplot(x='Group', y='Measure', data=melted_data, showfliers=False, ax=axes[i],
                    fill=False, linewidth=1, color='black', width=0.5,
                    medianprops=dict(color='gray', linewidth=2))
        sns.stripplot(x='Group', y='Measure', data=melted_data, ax=axes[i],
                      color="black", size=3)
        axes[i].grid(axis='y')
        sns.despine(bottom=True, left=True)
        axes[i].tick_params(left=False, bottom=False)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].set_title(ylabels[i])

    # for ax in axes:
        # ax.grid(True, which='both', axis='y', linestyle='-', linewidth=2)
        # ax.minorticks_on()
        # ax.grid(True, which='minor', axis='y', linestyle='-', linewidth=0.5)
        # ax.tick_params(axis='both', which='both', length=0)

    plt.suptitle(f"Dominant vs Nondominant extremity {ylabels[-1]} measures")
    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

pairs = [
    ('YBT_ANT_DOMINANT', 'YBT_ANT_NONDOMINANT'),
    ('YBT_PM_DOMINANT', 'YBT_PM_NONDOMINANT'),
    ('YBT_PL_DOMINANT', 'YBT_PL_NONDOMINANT'),
    ('YBT_COMPOSITE_DOMINANT', 'YBT_COMPOSITE_NONDOMINANT')]

ylabels = ['YBT ANT [%]', 'YBT PM [%]', 'YBT PL [%]', 'YBT COMP [%]', 'YBT']

draw_box_plots(data_YBT, pairs, ylabels)

pairs = [
    ('HAbd_DOMINANT_PEAK_FORCE', 'HAbd_NONDOMINANT_PEAK_FORCE'),
    ('KE_DOMINANT_PEAK_FORCE', 'KE_NONDOMINANT_PEAK_FORCE'),
    ('KF_DOMINANT_PEAK_FORCE', 'KF_NONDOMINANT_PEAK_FORCE'),
    ('AF_DOMINANT_PEAK_FORCE', 'AF_NONDOMINANT_PEAK_FORCE')
]
ylabels = ['HHD HAbd [kg]', 'HHD KE [kg]', 'HHD KF [kg]', 'HHD AF [kg]', 'HHD']
draw_box_plots(data_HHD, pairs, ylabels)

# %%
