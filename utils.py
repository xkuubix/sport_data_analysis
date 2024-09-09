import os
import pandas as pd
import logging
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_data(data_path, file_name):
    logger.info('Loading data')
    default_missing = pd._libs.parsers.STR_NA_VALUES
    default_missing = default_missing.remove('')
    data = pd.read_csv(os.path.join(data_path, file_name),
                       delimiter=';', encoding='utf-8',
                       na_filter=True, na_values=default_missing)
    return data


def rename_columns(data):
    logger.info('Renaming columns')
    columns_before = [col for col in data.columns]
    # Sports
    data.rename(columns={'Sports (1 = individual, 2 = team sports)': 'Sports'}, inplace=True)
    data.rename(columns={'Sports_Specialization ':'Sports_Specialization'}, inplace=True)
    data.rename(columns={'Sports_Specialization (points) (0-1 low, 2 moderate, 3 high)': 'Sports_Specialization_ordinal'}, inplace=True)
    data.rename(columns={'Experience_main_sport (years)': 'Experience_main_sport'}, inplace=True)
    data.rename(columns={'Training_Volume_Weekly_MainSport (hrs)': 'Training_Volume_Weekly_MainSport'}, inplace=True)
    data.rename(columns={'Training_Volume_Weekly_ALLSports (hrs)': 'Training_Volume_Weekly_ALLSports'}, inplace=True)

    # Questions
    data.rename(columns={'Have you given up a sport for your main sport?': 'Given_up_sport_for_main'}, inplace=True)
    data.rename(columns={'Have you trained in a main sport for more than 8 months in one year?': 'Months_in_a_year>8'}, inplace=True)
    data.rename(columns={'Athletes who participated in their primary sport for more hours per week than their age (Yes/No)': 'Hours_per_week>Age'}, inplace=True)
    data.rename(columns={'Is your main sport significantly more important than other sports?': 'Main_sport_more_important'}, inplace=True)
    data.rename(columns={'At what age did you start your main sport? (years)': 'Age_started_main_sport'}, inplace=True)

    # Injury
    data.rename(columns={'Injury_History (yes = 1, no = 0)': 'Injury_History'}, inplace=True)
    data.rename(columns={'Pain_now (0=NO, 1=YES)': 'Pain_now'}, inplace=True)

    # Demographics
    data.rename(columns={'Sex_(M=1, F=2)': 'Sex'}, inplace=True)
    data.rename(columns={'QoL - EQ-5D-Y': 'QoL_EQ-5D-Y'}, inplace=True)
    data.rename(columns={'Maturity_Offset (years)': 'Maturity_Offset'}, inplace=True)

    # Sport performance
    data.rename(columns={'PŚ_L PEAK FORCE (KG) Normalized to body weight Raw': 'PS_L_PEAK_FORCE'}, inplace=True)
    data.rename(columns={'PŚ_P PEAK FORCE (KG) Normalized to body weight Raw': 'PS_R_PEAK_FORCE'}, inplace=True)
    data.rename(columns={'CZ_L PEAK FORCE (KG) Normalized to body weight Raw': 'CZ_L_PEAK_FORCE'}, inplace=True)
    data.rename(columns={'CZ_P PEAK FORCE (KG) Normalized to body weight Raw': 'CZ_R_PEAK_FORCE'}, inplace=True)
    data.rename(columns={'DW_L PEAK FORCE(KG) Normalized to body weight Raw': 'CZ_L_PEAK_FORCE'}, inplace=True)
    data.rename(columns={'DW_P PEAK FORCE (KG) Normalized to body weight Raw': 'CZ_R_PEAK_FORCE'}, inplace=True)
    data.rename(columns={'BR_L PEAK FORCE (KG) Normalized to body weight Raw': 'BR_L_PEAK_FORCE'}, inplace=True)
    data.rename(columns={'BR_P_PEAK_FORCE_(KG)_Normalized_to_body_weight Raw': 'BR_R_PEAK_FORCE'}, inplace=True)
    columns_after = [col for col in data.columns]
    for i, (before, after) in enumerate(zip(columns_before, columns_after)):
        if before != after:
            if len(before) > 55:
                before = before[:50] + '...'
            logger.info(f'\t{before:55} >> {after:55}')
    return data


def clean_data(data):
    logger.info('Cleaning data')
    # Normalize binary entries
    data['Sports'] = data['Sports'].map({1: 'individual', 2: 'team'})
    data['Given_up_sport_for_main'] = data['Given_up_sport_for_main'].map(lambda x: 'Yes' if x.lower().startswith('tak') else 'No', na_action='ignore')
    data['Main_sport_more_important'] = data['Main_sport_more_important'].map(lambda x: 'Yes' if x.lower().startswith('tak') else 'No', na_action='ignore')
    data['Months_in_a_year>8'] = data['Months_in_a_year>8'].map(lambda x: 'Yes' if x.lower().startswith('tak') else 'No', na_action='ignore')
    data['Injury_History'] = data['Injury_History'].map({1: 'Yes', 0: 'No'})
    # Correct error made in data collection
    data['Sport'] = data['Sport'].str.lower()
    data['Sport'] = data['Sport'].str.replace('teakwondo', 'taekwondo')
    return data


def correct_ambiguity(data):
    logger.info('Correcting ambiguities')
    # drop rows with ambiguous injury information
    if 'Injury_History' in data.columns and 'Injury_History_Localization_Upper_Lower_Torso' in data.columns:
        data = data[((data["Injury_History"] == "No") & (data["Injury_History_Localization_Upper_Lower_Torso"] == "") & (data["Injury_History_Overuse_Acute"] == "")) | ((data["Injury_History"] == "Yes") & (data["Injury_History_Overuse_Acute"] != "") & (data["Injury_History_Overuse_Acute"] != ""))]
    if 'Training_Volume_Weekly_ALLSports' in data.columns and 'Training_Volume_Weekly_MainSport' in data.columns:
        data.loc[data["Training_Volume_Weekly_ALLSports"] == '', "Training_Volume_Weekly_ALLSports"] = data.loc[data["Training_Volume_Weekly_ALLSports"] == '', "Training_Volume_Weekly_MainSport"]
    return data

def erase_rows(data):
    logger.info('Erasing rows head/neck injuries')
    if 'Injury_History_Localization_Upper_Lower_Torso' in data.columns:
        data = data[data.Injury_History_Localization_Upper_Lower_Torso != "Head"]
        data = data[data.Injury_History_Localization_Upper_Lower_Torso != "Neck"]
    if 'Maturity_Offset' in data.columns:
        data = data[data['Maturity_Offset'] != "A"] # probably a mistake (NA)

    return data


def correct_dtype(data):
    logger.info('Correcting data types object -> float')
    if 'Training_Volume_Weekly_ALLSports' in data.columns:
        data["Training_Volume_Weekly_ALLSports"] = data["Training_Volume_Weekly_ALLSports"].astype(float)
    if 'Training_Volume_Weekly_MainSport' in data.columns:
        data["Training_Volume_Weekly_MainSport"] = data["Training_Volume_Weekly_MainSport"].astype(float)
    if 'Age_started_main_sport' in data.columns:
        data["Age_started_main_sport"] = data["Age_started_main_sport"].astype(float)
    if 'Chronologic_Age' in data.columns:
        data["Chronologic_Age"] = data["Chronologic_Age"].astype(float)
    if 'Experience_main_sport' in data.columns:
        data["Experience_main_sport"] = data["Experience_main_sport"].astype(float)
    if 'Maturity_Offset' in data.columns:
        data["Maturity_Offset"] = data["Maturity_Offset"].astype(float)
    if 'Sports_Specialization_ordinal' in data.columns:
        data["Sports_Specialization_ordinal"] = data["Sports_Specialization_ordinal"].astype(float)
    return data


def recover_missing(data, missing_values=None):
    number_of_missing = data.notna().all(axis=1).sum()
    if missing_values == 'BMI':
        if 'BMI' not in data.columns or 'Height (cm)' not in data.columns or 'Weight (kg)' not in data.columns:
            logger.error('Missing columns: BMI, Height (cm), Weight (kg)')
            return data
        logger.info(f'Recovering missing {missing_values} values, current number of missing values: %d', number_of_missing)
        # if we have empty BMI, we can calculate it with formula: weight / (height^2)
        # if we have empty height, we can calculate it with formula: sqrt(weight / BMI) * 100
        # if we have empty weight, we can calculate it with formula: BMI * (height^2)
        data.loc[data['BMI'].isna(), 'BMI'] = data['Weight (kg)'] / ((data['Height (cm)'] / 100) ** 2)
        data.loc[data['Height (cm)'].isna(), 'Height (cm)'] = np.sqrt(data.loc[data['Height (cm)'].isna(), 'Weight (kg)'] / data.loc[data['Height (cm)'].isna(), 'BMI']) * 100
        data.loc[data['Weight (kg)'].isna(), 'Weight (kg)'] = data.loc[data['Weight (kg)'].isna(), 'BMI'] * ((data.loc[data['Weight (kg)'].isna(), 'Height (cm)'] / 100) ** 2)
    if missing_values == 'PHV':
        if 'Age_PHV' not in data.columns:
            logger.error('Missing column: Age_PHV')
            return data
        logger.info(f'Recovering missing {missing_values} values, current number of missing values: %d', number_of_missing)
        # age phv calculation
        data["Leg_length (cm)"] = data["Height (cm)"] - data["Sitting_Height (cm)"]
        data["Leg_length_sitting_height_interaction"] = data["Leg_length (cm)"] * data["Sitting_Height (cm)"]
        data["Age_leg_Length_interaction"] = data["Leg_length (cm)"] * data["Chronologic_Age"]
        data["Age_sitting_height_interaction"] = data["Chronologic_Age"] * data["Sitting_Height (cm)"]
        data["Age_weight_interaction"] = data["Chronologic_Age"] * data["Weight (kg)"]
        data['Weight_height_ratio'] = data['Weight (kg)'] / data['Height (cm)'] * 100
        data["Sitting_Standing_Height_Ratio"] = data["Sitting_Height (cm)"] / data["Height (cm)"]
        data["Maturity_Offset_calculated"] = np.where(data["Sex"] == 2,
                                                    (-9.376 + (0.0001882 * data["Leg_length_sitting_height_interaction"]) + (0.0022 * data["Age_leg_Length_interaction"]) + (0.005841 * data["Age_sitting_height_interaction"]) + (-0.002658 * data["Age_weight_interaction"]) + (0.07693 * data["Weight_height_ratio"])),
                                                    (-9.3236 + (0.0002708 * data["Leg_length_sitting_height_interaction"]) + (-0.001663 * data["Age_leg_Length_interaction"]) + (0.007216 * data["Age_sitting_height_interaction"]) + (0.02292 * data["Weight_height_ratio"])))
        data["Age_PHV_calculated"] = data["Chronologic_Age"] - data["Maturity_Offset_calculated"]
    number_of_missing = data.notna().all(axis=1).sum()
    logger.info('Number of missing values after recovery: %d', number_of_missing)
    return data


def print_columns(data):
    for col in data.columns:
        print(col)
    return data