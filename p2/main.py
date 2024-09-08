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
    'CZ_L_PEAK_FORCE',
    'CZ_R_PEAK_FORCE',
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
    # d = d[d['Pain_now'] == 0] # do not consider people with pain
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
venn = venn3(subsets=lengths, set_labels=('HDD', 'YBT', 'FMS'))
plt.title('Liczba "pełnych" sportowców w poszczególnych testach/kombinacjach')
plt.show()
# %%
