import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import pyarrow as pa
import pyarrow.parquet as pq
import gcsfs
import getpass
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
import geopandas as gpd
from scikeras.wrappers import KerasRegressor
import tensorflow as tf
import matplotlib.pyplot as plt
# import sgis as sg
import dapla as dp
import datetime
from dapla.auth import AuthClient
from dapla import FileClient
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import requests
from pyjstat import pyjstat
# import plotly.express as px
from ipywidgets import interact, Dropdown
# from klass import search_classification

import sys

sys.path.append("../functions")
import kommune_pop
import kommune_inntekt
import kpi
import ao
import kommune_translate
import kommune
import ml_modeller

fs = FileClient.get_gcs_file_system()
import numpy as np


import warnings

warnings.filterwarnings("ignore")

import math

# good_df = ao.rette_bedrifter(good_df)

import input_data
# import create_datafiles

from joblib import Parallel, delayed
import multiprocessing

import time
import create_datafiles





def create_bedrift_fil(year, model, rate, scaler, GridSearch=False):

    start_time = time.time()
    
    current_year_good_oms, current_year_bad_oms, v_orgnr_list_for_imputering, training_data, imputatable_df, time_series_df = create_datafiles.main(year, rate)
    
    # Construct the function name dynamically
    function_name = f"{model}"
    # Use getattr to get the function from the oppdateringsfil module
    function_to_call = getattr(ml_modeller, function_name)
    
    # imputed_df = ml_modeller.knn_model(training_data, scaler, imputatable_df, GridSearch=False)
    
    imputed_df = function_to_call(training_data, scaler, imputatable_df, GridSearch=GridSearch)
    
    df_to_merge = imputed_df[['v_orgnr', 'year', 'id', 'predicted_oms']]
    
    bad_df = pd.merge(current_year_bad_oms, df_to_merge, on=['v_orgnr', 'id', 'year'], how='left')
    
    bad_df['new_oms'] = bad_df['predicted_oms']

    bad_df.drop(['predicted_oms'], axis=1, inplace=True)

    good_df = pd.concat([current_year_good_oms, bad_df], ignore_index=True)
    
    good_df = good_df[good_df['lopenr'] == 1]
    
    # if 'new_oms' is less than 0 then 'new_oms' = 0
    good_df['new_oms'] = good_df['new_oms'].apply(lambda x: 0 if x < 0 else x)

    good_df.drop(['tot_oms_fordelt'], axis=1, inplace=True)
    
    # Group by 'id' and calculate the sum
    grouped = (
        good_df.groupby("id")[["new_oms"]].sum().reset_index()
    )
    
    # Rename the columns
    grouped.rename(
        columns={"new_oms": "tot_oms_fordelt"},
        inplace=True,
    )
    
    good_df = pd.merge(good_df, grouped, on="id", how="left")
    
    good_df['id'] = good_df['id'].astype(str)
    good_df['nacef_5'] = good_df['nacef_5'].astype(str)
    good_df['orgnr_n_1'] = good_df['orgnr_n_1'].astype(str)
    good_df['b_kommunenr'] = good_df['b_kommunenr'].astype(str)
    good_df['forbruk'] = good_df['forbruk'].astype(float)
    good_df['salgsint'] = good_df['salgsint'].astype(float)
    good_df['tmp_no_p4005'] = good_df['tmp_no_p4005'].astype(float)
    
#     good_df["driftsk"] = good_df["gjeldende_driftsk_kr"]

#     # Convert columns to numeric
#     good_df["tot_driftskost_fordelt"] = pd.to_numeric(
#         good_df["tot_driftskost_fordelt"], errors="coerce"
#     )
#     good_df["driftsk"] = pd.to_numeric(good_df["driftsk"], errors="coerce")


#     good_df["drkost_share"] = good_df["driftsk"] / good_df["tot_driftskost_fordelt"]

#     good_df['drkost_share'].fillna(0, inplace=True)

#     good_df["new_drkost"] = good_df["drkost_share"] * good_df["foretak_driftskostnad"]

#     # if new_drkost is NaN then new_drkost = gjeldende_driftsk_kr
#     good_df["new_drkost"].replace([np.inf, -np.inf], np.nan, inplace=True)
#     good_df["new_drkost"].fillna(good_df["gjeldende_driftsk_kr"], inplace=True)
    
#     good_df.drop(['tot_driftskost_fordelt', 'drkost_share'], axis=1, inplace=True)
    
#     good_df['new_drkost'] = good_df['new_drkost'].astype(float)
    
#     grouped = (
#         good_df.groupby("id")[["new_drkost"]].sum().reset_index()
#     )

#     grouped.rename(
#         columns={"new_drkost": "tot_driftskost_fordelt"},
#         inplace=True,
#     )

#     good_df = pd.merge(good_df, grouped, on="id", how="left")

#     good_df.sort_values(by="id", ascending=True, inplace=True)

#     good_df['drkost_share'] = good_df['new_drkost']/good_df['tot_driftskost_fordelt']

#     good_df["drkost_share"].replace([np.inf, -np.inf], np.nan, inplace=True)
#     good_df['drkost_share'].fillna(0, inplace=True)


#     good_df['new_drkost'] = good_df['drkost_share'] * good_df['foretak_driftskostnad']

    ### Ny metode
    
    condition = (good_df['foretak_omsetning'] == 0) | (good_df['foretak_driftskostnad'] == 0)

    # Drop the rows that meet the condition
    good_df = good_df[~condition]

    good_df["gjeldende_driftsk_kr"] = pd.to_numeric(good_df["gjeldende_driftsk_kr"], errors="coerce")
    good_df["b_sysselsetting_syss"] = pd.to_numeric(good_df["b_sysselsetting_syss"], errors="coerce")

    good_df.drop(['tot_driftskost_fordelt'], axis=1, inplace=True)

    good_df["driftsk"] = good_df["gjeldende_driftsk_kr"]

    grouped = (
        good_df.groupby("id")[["driftsk"]].sum().reset_index()
    )


    grouped.rename(
        columns={"driftsk": "tot_driftskost_fordelt"},
        inplace=True,
    )

    good_df = pd.merge(good_df, grouped, on="id", how="left")

    # Convert columns to numeric
    good_df["tot_driftskost_fordelt"] = pd.to_numeric(
        good_df["tot_driftskost_fordelt"], errors="coerce"
    )
    good_df["driftsk"] = pd.to_numeric(good_df["driftsk"], errors="coerce")

    good_df["b_sysselsetting_syss"] = pd.to_numeric(good_df["b_sysselsetting_syss"], errors="coerce")

    # Convert 'lonn' to numeric, replacing comma with dot and handling errors
    good_df["lonn"] = good_df["lonn"].str.replace(',', '.').astype(float)
    good_df["lonn"] = good_df["lonn"] / 100

    # Calculate drkost_share
    # Calculate drkost_share
    good_df["drkost_share"] = good_df.apply(
        lambda row: row["lonn"] if row["foretak_driftskostnad"] != 0 and row["tot_driftskost_fordelt"] == 0 else (row["driftsk"] / row["tot_driftskost_fordelt"] if row["tot_driftskost_fordelt"] != 0 else np.nan),
        axis=1
    )

    # Handle any NaN or inf values in drkost_share
    good_df['drkost_share'].replace([np.inf, -np.inf], np.nan, inplace=True)
    good_df['drkost_share'].fillna(0, inplace=True)

    # Calculate total b_sysselsetting_syss per id
    good_df['total_syss'] = good_df.groupby('id')['b_sysselsetting_syss'].transform('sum')

    good_df["total_syss"] = pd.to_numeric(good_df["total_syss"], errors="coerce")

    # Calculate the share of b_sysselsetting_syss per id
    good_df['syss_share'] = good_df['b_sysselsetting_syss'] / good_df['total_syss']

    # Update drkost_share for the specified conditions
    good_df.loc[
        (good_df['tot_driftskost_fordelt'] == 0) & 
        (good_df['drkost_share'] == 0) & 
        (good_df['foretak_driftskostnad'] != 0), 
        'drkost_share'
    ] = good_df['syss_share']

    # Round drkost_share to 10 decimal points
    good_df['drkost_share'] = good_df['drkost_share'].round(10)


    # Calculate new_drkost
    good_df["new_drkost"] = good_df["drkost_share"] * good_df["foretak_driftskostnad"]

    # Replace NaN in new_drkost with gjeldende_driftsk_kr
    good_df["new_drkost"].fillna(good_df["gjeldende_driftsk_kr"], inplace=True)
    good_df['new_drkost'] = good_df['new_drkost'].astype(float)

    # if foretak_driftskostnad = 0 then new_drkost = 0
    good_df.loc[good_df['foretak_driftskostnad'] == 0, 'new_drkost'] = 0

    # if foretak_omsetning = 0 then new_oms = 0
    good_df.loc[good_df['foretak_omsetning'] == 0, 'new_oms'] = 0

    ## end ny metode

    good_df['new_drkost'] = good_df['new_drkost'].astype(float)

    good_df.drop(['drkost_share'], axis=1, inplace=True)
    
    # old code
    
#     mask_regtype_04 = good_df['regtype'] == '04'

#     # Step 2: Update new_oms where regtype is '04'
#     good_df.loc[mask_regtype_04, 'new_oms'] = good_df.loc[mask_regtype_04, 'new_drkost']
    
#     # Step 1: Create a boolean mask for regtype = '04'
#     mask_regtype_04 = good_df['regtype'] == '04'

#     # Step 2: Group by 'id' and sum 'new_oms' where regtype is '04'
#     total_helper_oms = good_df.loc[mask_regtype_04].groupby('id')['new_oms'].sum().reset_index()
    
#     # Rename the aggregated 'new_oms' to 'new_oms_total_helper' for clarity
#     total_helper_oms.rename(columns={'new_oms': 'new_oms_total_helper'}, inplace=True)

#     # Step 3: Merge the aggregated result back into the original DataFrame
#     good_df = pd.merge(good_df, total_helper_oms, on='id', how='left', suffixes=('', '_total_helper'))
    
#     # Reset index to ensure it's unique
#     good_df.reset_index(drop=True, inplace=True)
    
#     good_df['foretak_omsetning'] = pd.to_numeric(good_df['foretak_omsetning'], errors='coerce')
    
#     good_df['foretak_omsetning'].fillna(0, inplace=True)
#     good_df['new_oms_total_helper'].fillna(0, inplace=True)

#     good_df['new_oms_total_helper'] = pd.to_numeric(good_df['new_oms_total_helper'], errors='coerce')
    
#     good_df['total_rest_oms'] = good_df['foretak_omsetning'] - good_df['new_oms_total_helper']
    
#     # Step 1: Create a boolean mask for regtype != '04'
#     mask_regtype_not_04 = good_df['regtype'] != '04'

#     # Step 2: Group by 'id' and sum 'new_oms' where regtype is not '04'
#     total_non_helper_oms = good_df.loc[mask_regtype_not_04].groupby('id')['new_oms'].sum().reset_index()

#     # Step 3: Merge the aggregated result back into the original DataFrame
#     good_df = pd.merge(good_df, total_non_helper_oms, on='id', how='left', suffixes=('', '_total_non_helper'))

#     good_df['new_oms_total_non_helper'].fillna(0, inplace=True)
    
#     # Convert 'new_oms' and 'new_oms_total_non_helper' to numeric, coercing errors to NaN
#     good_df['new_oms'] = pd.to_numeric(good_df['new_oms'], errors='coerce')
#     good_df['new_oms_total_non_helper'] = pd.to_numeric(good_df['new_oms_total_non_helper'], errors='coerce')

#     # Replace zeros in 'new_oms_total_non_helper' to avoid division by zero
#     good_df['new_oms_total_non_helper'].replace(0, np.nan, inplace=True)

#     # Perform safe division
#     good_df['oms_share_non_helpers'] = good_df['new_oms'] / good_df['new_oms_total_non_helper']

#     # Set the oms_share_non_helpers to 0 where regtype is '04'
#     good_df.loc[good_df['regtype'] == '04', 'oms_share_non_helpers'] = 0

#     # Replace NaN in 'oms_share_non_helpers' resulting from division by zero or NaN in the denominator
#     good_df['oms_share_non_helpers'].fillna(0, inplace=True)

#     # Optionally, replace NaN resulting from any remaining errors with a default value
#     good_df.fillna(0, inplace=True)
    
#     # Step 1: Use numpy.where to conditionally update new_oms
#     good_df['new_oms'] = np.where(good_df['regtype'] == '04', 
#                                   good_df['new_oms'], 
#                                   good_df['oms_share_non_helpers'] * good_df['total_rest_oms'])


    # new code:
    
    good_df.drop(['tot_oms_fordelt'], axis=1, inplace=True)
    
    grouped = (
        good_df.groupby("id")[["new_oms"]].sum().reset_index()
    )

    grouped.rename(
        columns={"new_oms": "tot_oms_fordelt"},
        inplace=True,
    )

    good_df = pd.merge(good_df, grouped, on="id", how="left")

    mask_regtype_04 = good_df['regtype'] == '04'
    mask_regtype_not_04 = good_df['regtype'] != '04'

    # Step 2: Update new_oms where regtype is '04'
    good_df.loc[mask_regtype_04, 'new_oms'] = good_df.loc[mask_regtype_04, 'new_drkost']

    # Step 2: Group by 'id' and sum 'new_oms' where regtype is '04'
    total_helper_oms = good_df.loc[mask_regtype_04].groupby('id')['new_oms'].sum().reset_index()



    # Rename the aggregated 'new_oms' to 'new_oms_total_helper' for clarity
    total_helper_oms.rename(columns={'new_oms': 'new_oms_total_helper'}, inplace=True)

    # Step 3: Merge the aggregated result back into the original DataFrame
    good_df = pd.merge(good_df, total_helper_oms, on='id', how='left', suffixes=('', '_total_helper'))

    # Reset index to ensure it's unique
    good_df.reset_index(drop=True, inplace=True)

    # Convert 'foretak_omsetning' to numeric, setting errors to NaN
    good_df['foretak_omsetning'] = pd.to_numeric(good_df['foretak_omsetning'], errors='coerce')

    # Fill NaN values that might have been introduced by conversion errors
    good_df['foretak_omsetning'].fillna(0, inplace=True)
    good_df['new_oms_total_helper'].fillna(0, inplace=True)

    good_df['new_oms_total_helper'] = pd.to_numeric(good_df['new_oms_total_helper'], errors='coerce')

    # Step 4: Subtract 'new_oms_total_helper' from 'foretak_omsetning'
    good_df['total_rest_oms'] = good_df['foretak_omsetning'] - good_df['new_oms_total_helper']

    # Step 2: Group by 'id' and sum 'new_oms' where regtype is not '04'
    total_non_helper_oms = good_df.loc[mask_regtype_not_04].groupby('id')['new_oms'].sum().reset_index()

    # Step 3: Merge the aggregated result back into the original DataFrame
    good_df = pd.merge(good_df, total_non_helper_oms, on='id', how='left', suffixes=('', '_total_non_helper'))

    good_df['new_oms_total_non_helper'].fillna(0, inplace=True)



    # test 4

    # Convert 'new_oms' and 'new_oms_total_non_helper' to numeric, coercing errors to NaN
    good_df['new_oms'] = pd.to_numeric(good_df['new_oms'], errors='coerce')
    good_df['new_oms_total_non_helper'] = pd.to_numeric(good_df['new_oms_total_non_helper'], errors='coerce')

    # convert Nan in new_oms_total_non_helper to 0
    good_df['new_oms_total_non_helper'].fillna(0, inplace=True)

    ## Here we need to add some stuff
    # Calculate total lonn per id excluding regtype '04'
    good_df['total_lonn_non_04'] = good_df[mask_regtype_not_04].groupby('id')['lonn'].transform('sum')

    # Ensure total_lonn_non_04 is numeric and handle any conversion issues
    good_df['total_lonn_non_04'] = pd.to_numeric(good_df['total_lonn_non_04'], errors='coerce')

    # Recalculate lonn for non-'04' rows to sum to 100% per id
    good_df['lonn_non_04_share'] = np.where(
        mask_regtype_not_04,
        good_df['lonn'] / good_df['total_lonn_non_04'],
        0
    )

    good_df['lonn_non_04_share'].fillna(0, inplace=True)

    # Calculate total b_sysselsetting_syss per id excluding regtype '04'
    good_df['total_syss_non_04'] = good_df[mask_regtype_not_04].groupby('id')['b_sysselsetting_syss'].transform('sum')

    # Calculate the share of b_sysselsetting_syss excluding regtype '04'
    good_df['syss_share_non_04'] = good_df['b_sysselsetting_syss'] / good_df['total_syss_non_04']

    good_df['syss_share_non_04'].fillna(0, inplace=True)

    # Calculate oms_share_non_helpers with the new condition
    good_df['oms_share_non_helpers'] = good_df.apply(
        lambda row: 1 if row['total_rest_oms'] == 0 else (
            row['lonn_non_04_share'] if row['foretak_omsetning'] != 0 and row['tot_oms_fordelt'] == 0 else (
                row["new_oms"] / row["new_oms_total_non_helper"] if row["new_oms_total_non_helper"] != 0 else row['syss_share_non_04'])),
        axis=1
    )


    ######################################

    # Handle any NaN or inf values in oms_share_non_helpers
    good_df['oms_share_non_helpers'].replace([np.inf, -np.inf], np.nan, inplace=True)
    good_df['oms_share_non_helpers'].fillna(0, inplace=True)

    # Ensure oms_share_non_helpers is set to 0 where regtype is '04'
    good_df.loc[good_df['regtype'] == '04', 'oms_share_non_helpers'] = 0

    # Finally, update new_oms with the calculated oms_share_non_helpers
    good_df['new_oms'] = np.where(good_df['regtype'] == '04', 
                                  good_df['new_oms'], 
                                  good_df['oms_share_non_helpers'] * good_df['total_rest_oms'])


    
    # end new code:
    
    # Define the values for w_naring_vh, w_nace1_ikke_vh, and w_nace2_ikke_vh
    w_naring_vh = ("45", "46", "47")
    w_nace1_ikke_vh = "45.403"
    w_nace2_ikke_vh = ("45.2", "46.1")

    enhetene_brukes = good_df.copy()
    
    del good_df

    # Filter the DataFrame based on conditions and create vhbed variable
    enhetene_brukes["vhbed"] = 0

    # Check if the first two characters of 'naring' are in w_naring_vh
    enhetene_brukes.loc[
        enhetene_brukes["tmp_sn2007_5"].str[:2].isin(w_naring_vh), "vhbed"
    ] = 1

    # Check if 'naring' is in w_nace1_ikke_vh
    enhetene_brukes.loc[enhetene_brukes["tmp_sn2007_5"] == w_nace1_ikke_vh, "vhbed"] = 0

    # Check if the first four characters of 'naring' are in w_nace2_ikke_vh
    enhetene_brukes.loc[
        enhetene_brukes["tmp_sn2007_5"].str[:4].isin(w_nace2_ikke_vh), "vhbed"
    ] = 0
    
    enhetene_brukes = enhetene_brukes.drop_duplicates(subset=["orgnr_n_1", "lopenr", "radnr", "v_orgnr"])
    
    enhetene_brukes ['check'] = enhetene_brukes["foretak_driftskostnad"] - enhetene_brukes["forbruk"]
    
    enhetene_brukes ['forbruk'] = np.where(
        enhetene_brukes['check'] < 0, enhetene_brukes['forbruk'] / 1000, enhetene_brukes['forbruk']
    )
    
    # 2
    salgsint_forbruk = enhetene_brukes[
        [
            "orgnr_n_1",
            "lopenr",
            "v_orgnr",
            "forbruk",
            "salgsint",
            "radnr",
            "nacef_5",
            "tmp_sn2007_5",
            "new_oms",
            "vhbed",
        ]
    ]
    
    # 3

    har = salgsint_forbruk[salgsint_forbruk.groupby("orgnr_n_1")["vhbed"].transform("any")]
    # Extract the 'orgnr_n_1' column
    har = har[["orgnr_n_1"]]

    # Remove duplicates
    har.drop_duplicates(inplace=True)
    
    ikke_har = salgsint_forbruk[
        ~salgsint_forbruk.groupby("orgnr_n_1")["vhbed"].transform("any")
    ]
    ikke_har = ikke_har[["orgnr_n_1"]]
    ikke_har.drop_duplicates(inplace=True)

    ikke_har["ikkevbed"] = 1
    
    # 5


    # Merge ikke_har into salgsint_forbruk with a left join on the 'id' column
    salgsint_forbruk_update1 = pd.merge(
        salgsint_forbruk, ikke_har, on="orgnr_n_1", how="left"
    )

    # salgsint_forbruk_update1['ikkevbed'].fillna(0, inplace=True)

    # Update 'vhbed' to 1 where 'ikkevbed' is 1
    salgsint_forbruk_update1.loc[salgsint_forbruk_update1["ikkevbed"] == 1, "vhbed"] = 1
    
    # 6
    # Assuming your original DataFrame is named salgsint_forbruk_update1
    # Replace 'new_oms', 'orgnr_foretak', 'lopnr', 'vhbed' with the actual column names in your DataFrame

    # Create sum1 DataFrame for vhbed=1
    sum1 = (
        salgsint_forbruk_update1[salgsint_forbruk_update1["vhbed"] == 1]
        .groupby(["orgnr_n_1", "lopenr"])["new_oms"]
        .sum()
        .reset_index()
    )
    sum1.rename(columns={"new_oms": "sumoms_vh"}, inplace=True)

    # Create sum2 DataFrame for vhbed=0
    sum2 = (
        salgsint_forbruk_update1[salgsint_forbruk_update1["vhbed"] == 0]
        .groupby(["orgnr_n_1", "lopenr"])["new_oms"]
        .sum()
        .reset_index()
    )
    sum2.rename(columns={"new_oms": "sumoms_andre"}, inplace=True)
    
    sum3 = pd.merge(sum1, sum2, on=["orgnr_n_1", "lopenr"], how="outer")
    
    salgsint_forbruk_update2 = pd.merge(
        salgsint_forbruk_update1, sum3, on=["orgnr_n_1", "lopenr"], how="outer"
    )
    
    # Sort the DataFrame by 'orgnr_n_1', 'lopenr', and 'rad_nr'
    salgsint_forbruk_update2.sort_values(by=["orgnr_n_1", "lopenr", "radnr"], inplace=True)

    salgsint_forbruk_update2.sort_values(by=["orgnr_n_1", "lopenr", "vhbed"], inplace=True)
    
    # Sort the DataFrame by 'orgnr_foretak' and 'lopenr'

    salgsint_forbruk_update3 = salgsint_forbruk_update2.copy()

    salgsint_forbruk_update3.sort_values(by=["orgnr_n_1", "lopenr"], inplace=True)

    # Create a new variable 'vhf' based on the values of 'vhbed'
    salgsint_forbruk_update3["vhf"] = salgsint_forbruk_update3.groupby(
        ["orgnr_n_1", "lopenr"]
    )["vhbed"].transform("first")

    # Retain the value of 'vhf' from the first observation in each group
    salgsint_forbruk_update3["vhf"] = salgsint_forbruk_update3.groupby(
        ["orgnr_n_1", "lopenr"]
    )["vhf"].transform("first")

    # Apply labels to the variables
    salgsint_forbruk_update3["vhbed"] = salgsint_forbruk_update3["vhbed"].astype(str)
    salgsint_forbruk_update3["vhf"] = salgsint_forbruk_update3["vhf"].astype(str)

    label_map_vhbed = {"1": "varehandelsbedrift", "0": "annen type bedrift"}
    label_map_vhf = {
        "1": "foretaket har kun varehandelsbedrifter eller ingen",
        "0": "har varehandel og annen bedrift (blandingsnæringer)",
    }

    salgsint_forbruk_update3["vhbed"] = salgsint_forbruk_update3["vhbed"].map(
        label_map_vhbed
    )
    salgsint_forbruk_update3["vhf"] = salgsint_forbruk_update3["vhf"].map(label_map_vhf)
    
    # Filter rows where vhf is 'foretaket har kun varehandelsbedrifter eller ingen'
    vhf_condition = (
        salgsint_forbruk_update3["vhf"]
        == "foretaket har kun varehandelsbedrifter eller ingen"
    )
    vhf_df = salgsint_forbruk_update3.loc[vhf_condition]

    # Filter rows where vhf is not 'foretaket har kun varehandelsbedrifter eller ingen'
    andre_df = salgsint_forbruk_update3.loc[~vhf_condition]
    

    vhf_df["nokkel"] = vhf_df["new_oms"] / vhf_df["sumoms_vh"]

    # Convert 'salgsint' column to numeric
    vhf_df["salgsint"] = pd.to_numeric(vhf_df["salgsint"], errors="coerce")
    vhf_df["forbruk"] = pd.to_numeric(vhf_df["forbruk"], errors="coerce")


    vhf_df["bedr_salgsint"] = round(vhf_df["salgsint"] * vhf_df["nokkel"])
    vhf_df["bedr_forbruk"] = round(vhf_df["forbruk"] * vhf_df["nokkel"])
    
    # 13

    andre_df["forbruk"] = pd.to_numeric(andre_df["forbruk"], errors="coerce")
    andre_df["salgsint"] = pd.to_numeric(andre_df["salgsint"], errors="coerce")


    # Assuming 'andre' is your DataFrame
    andre_df["avanse"] = andre_df["forbruk"] / andre_df["salgsint"]

    # Filter rows where vhbed is 1
    vh_bedriftene = andre_df[andre_df["vhbed"] == "varehandelsbedrift"].copy()

    # Calculate 'nokkel', 'bedr_salgsint', and 'bedr_forbruk' for vh-bedriftene
    vh_bedriftene["nokkel"] = vh_bedriftene["new_oms"] / vh_bedriftene["sumoms_vh"]
    vh_bedriftene["bedr_salgsint"] = round(
        vh_bedriftene["salgsint"] * vh_bedriftene["nokkel"]
    )
    vh_bedriftene.loc[
        vh_bedriftene["bedr_salgsint"] > vh_bedriftene["new_oms"], "bedr_salgsint"
    ] = vh_bedriftene["new_oms"]
    vh_bedriftene["bedr_forbruk"] = round(
        vh_bedriftene["bedr_salgsint"] * vh_bedriftene["avanse"]
    )

    # Summarize vh-bedriftene
    brukt1 = (
        vh_bedriftene.groupby(["orgnr_n_1", "lopenr"])
        .agg({"bedr_salgsint": "sum", "bedr_forbruk": "sum"})
        .reset_index()
    )

    # Merge summarized values back to 'andre'
    andre = pd.merge(andre_df, brukt1, on=["orgnr_n_1", "lopenr"], how="left")

    # Calculate 'resten1' and 'resten2'
    andre["resten1"] = andre["salgsint"] - andre["bedr_salgsint"]
    andre["resten2"] = andre["forbruk"] - andre["bedr_forbruk"]

    # Filter rows where vhbed is not 1
    blanding_av_vh_og_andre = andre[andre["vhbed"] != "varehandelsbedrift"].copy()

    # Calculate 'nokkel', 'bedr_salgsint', and 'bedr_forbruk' for blending of vh and other industries
    blanding_av_vh_og_andre["nokkel"] = (
        blanding_av_vh_og_andre["new_oms"] / blanding_av_vh_og_andre["sumoms_andre"]
    )
    blanding_av_vh_og_andre["bedr_salgsint"] = round(
        blanding_av_vh_og_andre["resten1"] * blanding_av_vh_og_andre["nokkel"]
    )
    blanding_av_vh_og_andre["bedr_forbruk"] = round(
        blanding_av_vh_og_andre["resten2"] * blanding_av_vh_og_andre["nokkel"]
    )

    # Combine the two subsets back into 'andre'
    andre = pd.concat([vh_bedriftene, blanding_av_vh_og_andre], ignore_index=True)

    andre.sort_values(by=["orgnr_n_1", "lopenr"], inplace=True)

    oppdatere_hv = pd.concat([vhf_df, andre], ignore_index=True)

    oppdatere_hv = oppdatere_hv[
        ["orgnr_n_1", "lopenr", "radnr", "bedr_forbruk", "bedr_salgsint"]
    ]
    
    enhetene_brukes2 = pd.merge(
        enhetene_brukes, oppdatere_hv, on=["orgnr_n_1", "lopenr", "radnr"]
    )
    
    # Step 1: Identify IDs that appear more than once
    duplicate_ids = enhetene_brukes2['id'][enhetene_brukes2['id'].duplicated(keep=False)]

    # Step 2: Update regtype to '02' where regtype is '01' and the id appears more than once
    enhetene_brukes2.loc[(enhetene_brukes2['regtype'] == '01') & (enhetene_brukes2['id'].isin(duplicate_ids)), 'regtype'] = '02'

    enhetene_brukes2.loc[enhetene_brukes2['regtype'] == '01', 'new_oms'] = enhetene_brukes2['foretak_omsetning']
    enhetene_brukes2.loc[enhetene_brukes2['regtype'] == '01', 'new_drkost'] = enhetene_brukes2['foretak_driftskostnad']
    enhetene_brukes2.loc[enhetene_brukes2['regtype'] == '01', 'bedr_salgsint'] = enhetene_brukes2['salgsint']
    enhetene_brukes2.loc[enhetene_brukes2['regtype'] == '01', 'bedr_forbruk'] = enhetene_brukes2['forbruk']
    
#     rettes = enhetene_brukes2.copy()

#     rettes["oms"] = rettes["new_oms"]
#     rettes["driftsk"] = rettes["gjeldende_driftsk_kr"]
    
#     # Convert columns to numeric
#     rettes["tot_driftskost_fordelt"] = pd.to_numeric(
#         rettes["tot_driftskost_fordelt"], errors="coerce"
#     )
#     rettes["driftsk"] = pd.to_numeric(rettes["driftsk"], errors="coerce")




#     rettes["drkost_share"] = rettes["driftsk"] / rettes["tot_driftskost_fordelt"]
    
#     rettes["new_drkost"] = rettes["drkost_share"] * rettes["foretak_driftskostnad"]
#     rettes["new_drkost"] = rettes["new_drkost"].replace([np.inf, -np.inf], np.nan)

#     rettes["profit_ratio"] = rettes["foretak_driftskostnad"] / rettes["foretak_omsetning"]

#     # if tot_driftskost_fordelt = 0 and 'new_drkost' is NaN then new_drkost = "profit_ratio" * new_oms
#     rettes["new_drkost"] = np.where(
#         (rettes["tot_driftskost_fordelt"] == 0) & (rettes["new_drkost"].isna()),
#         rettes["profit_ratio"] * rettes["new_oms"],
#         rettes["new_drkost"],
#     )

#     rettes["new_drkost"].isna().sum()
    
    rettes2 = enhetene_brukes2.copy()
    
    del enhetene_brukes2
    
    rettes2["oms"] = rettes2["new_oms"]
    
    rettes2["driftsk"] = rettes2["gjeldende_driftsk_kr"]
    
    # Convert columns to numeric
    rettes2["tot_driftskost_fordelt"] = pd.to_numeric(
        rettes2["tot_driftskost_fordelt"], errors="coerce"
    )
    rettes2["driftsk"] = pd.to_numeric(rettes2["driftsk"], errors="coerce")
    
    rettes2["new_drkost"] = rettes2["new_drkost"].fillna(0)
    
    
    rettes2["drkost_temp"] = rettes2["new_drkost"]

    # Fill NaN in 'drkost_temp' with 0
    rettes2["drkost_temp"] = rettes2["drkost_temp"].fillna(0)
    
    rettes2["gjeldende_lonn_kr"] = rettes2["gjeldende_lonn_kr"].str.replace(",", ".")

    rettes2["gjeldende_lonn_kr"] = pd.to_numeric(
        rettes2["gjeldende_lonn_kr"], errors="coerce"
    ).fillna(0)
    rettes2["bedr_forbruk"] = pd.to_numeric(
        rettes2["bedr_forbruk"], errors="coerce"
    ).fillna(0)


    rettes2["lonn_+_forbruk"] = rettes2["gjeldende_lonn_kr"] + rettes2["bedr_forbruk"]

    # Perform the if operation
    condition = rettes2["drkost_temp"] < rettes2["lonn_+_forbruk"]
    rettes2["drkost_temp"] = np.where(
        condition, rettes2["lonn_+_forbruk"], rettes2["drkost_temp"]
    )
    rettes2["theif"] = np.where(condition, 1, 0)
    
    dkvars = rettes2[rettes2.groupby("orgnr_n_1")["theif"].transform("any")]
    
    # Calculate 'utskudd'
    dkvars["utskudd"] = (
        dkvars["new_drkost"] - dkvars["gjeldende_lonn_kr"] - dkvars["bedr_forbruk"]
    )
    dkvars["utskudd"] = abs(dkvars["utskudd"])

    # Keep selected columns
    columns_to_keep = [
        "orgnr_n_1",
        "lopenr",
        "radnr",
        "utskudd",
        "new_drkost",
        "drkost_temp",
        "theif",
        "gjeldende_lonn_kr",
        "bedr_forbruk",
    ]
    dkvars = dkvars[columns_to_keep]
    
    # Calculate sum of 'utskudd' grouped by 'orgnr_foretak', 'lopenr', and 'tyv'
    sum7b = dkvars.groupby(["orgnr_n_1", "lopenr", "theif"])["utskudd"].sum().reset_index()

    sum7b_transposed = sum7b.pivot(
        index=["orgnr_n_1", "lopenr"], columns="theif", values="utskudd"
    ).reset_index()

    # Rename columns as per SAS code
    sum7b_transposed.rename(columns={0: "thief0", 1: "thief1"}, inplace=True)

    sum7b_transposed = sum7b_transposed[["orgnr_n_1", "lopenr", "thief0", "thief1"]]
    
    dkvars_2 = pd.merge(dkvars, sum7b_transposed, on=["orgnr_n_1", "lopenr"], how="inner")
    
    # Apply conditional logic
    pd.set_option("display.float_format", "{:.2f}".format)
    dkvars_2["andel1"] = np.where(
        dkvars_2["theif"] == 0, dkvars_2["utskudd"] / dkvars_2["thief0"], np.nan
    )
    dkvars_2["andel2"] = np.where(
        dkvars_2["theif"] == 0, np.round(dkvars_2["andel1"] * dkvars_2["thief1"]), np.nan
    )
    # dkvars_2['new_drkost'] = np.where(dkvars_2['theif'] == 0, np.sum(dkvars_2['drkost_temp'] - dkvars_2['andel2'], axis=0), dkvars_2['drkost_temp'])
    dkvars_2["new_drkost"] = np.where(
        dkvars_2["theif"] == 0,
        dkvars_2["drkost_temp"] - dkvars_2["andel2"],
        dkvars_2["drkost_temp"],
    )

    # Keep selected columns
    columns_to_keep = ["orgnr_n_1", "lopenr", "radnr", "new_drkost"]
    dkvars_3 = dkvars_2[columns_to_keep]

    # dkvars_2.head(50)
    
    merged_df = pd.merge(rettes2, dkvars_3, how='left', left_on=['orgnr_n_1', 'lopenr', 'radnr'], right_on=['orgnr_n_1', 'lopenr', 'radnr'], suffixes=('', '_updated'))
    
    del rettes2, dkvars_3, dkvars_2

    merged_df['new_drkost'] = merged_df['new_drkost_updated'].combine_first(merged_df['new_drkost'])

    duplicate_ids = merged_df['id'][merged_df['id'].duplicated(keep=False)]

    # Step 2: Update regtype to '02' where regtype is '01' and the id appears more than once
    merged_df.loc[(merged_df['regtype'] == '01') & (merged_df['id'].isin(duplicate_ids)), 'regtype'] = '02'

    merged_df.loc[merged_df['regtype'] == '01', 'new_oms'] = merged_df['foretak_omsetning']
    merged_df.loc[merged_df['regtype'] == '01', 'new_drkost'] = merged_df['foretak_driftskostnad']
    merged_df.loc[merged_df['regtype'] == '01', 'bedr_salgsint'] = merged_df['salgsint']
    merged_df.loc[merged_df['regtype'] == '01', 'bedr_forbruk'] = merged_df['forbruk']

    test_grouped = (
        merged_df.groupby("id")[["new_drkost"]].sum().reset_index()
    )

    test_grouped.rename(
        columns={"new_drkost": "test_tot_drkost_fordelt"},
        inplace=True,
    )

    temp = pd.merge(merged_df, test_grouped, on="id", how="left")

    temp['drkost_diff'] = temp['foretak_driftskostnad'] - temp['test_tot_drkost_fordelt']

    temp = temp.sort_values(by='drkost_diff', ascending=True)

    mask = temp['drkost_diff'].abs() <= 1000

    # Create a new DataFrame with the rows to be excluded
    check_manually = merged_df[~mask]

    # Update the original DataFrame to keep only the rows where the absolute value of 'drkost_diff' is <= 1000
    merged_df = merged_df[mask]

#     # convert b_sysselsetting_syss  to int
#     merged_df["b_sysselsetting_syss"] = merged_df["b_sysselsetting_syss"].astype(int)
#     merged_df["fjor_driftskost_kr_t1"] = merged_df["fjor_driftskost_kr_t1"].astype(int)
#     merged_df["fjor_lonn_kr_t1"] = merged_df["fjor_lonn_kr_t1"].astype(int)
#     merged_df["fjor_omsetn_kr_t1"] = merged_df["fjor_omsetn_kr_t1"].astype(int)
#     merged_df["fjor_snittlonn_t1"] = merged_df["fjor_snittlonn_t1"].astype(int)
#     merged_df["fjor_snittoms_t1"] = merged_df["fjor_snittoms_t1"].astype(int)
#     merged_df["fjor_syssel_t1"] = merged_df["fjor_syssel_t1"].astype(int)
#     merged_df["tmp_forbruk_bed"] = merged_df["tmp_forbruk_bed"].astype(int)
#     merged_df["tmp_ny_bdr_syss"] = merged_df["tmp_ny_bdr_syss"].astype(int)
#     merged_df["tmp_salgsint_bed"] = merged_df["tmp_salgsint_bed"].astype(int)
#     merged_df["tmp_snittlonn"] = merged_df["tmp_snittlonn"].astype(int)
#     merged_df["tmp_snittoms"] = merged_df["tmp_snittoms"].astype(int)

    # merged_df["fjor_nace_b_t1"] = merged_df["fjor_nace_b_t1"].astype(str)
    # merged_df["regtype"] = merged_df["regtype"].astype(str)
    # merged_df["tmp_sn2007_5"] = merged_df["tmp_sn2007_5"].astype(str)

    oppdateringsfil = merged_df.copy()
    
    time_series_df["n3"] = time_series_df["tmp_sn2007_5"].str[:4]
    time_series_df["n2"] = time_series_df["tmp_sn2007_5"].str[:2]
    merged_df["n2"] = merged_df["tmp_sn2007_5"].str[:2]

    temp_1 = time_series_df[['id',
                             'nacef_5',
                             'orgnr_n_1',
                             'b_sysselsetting_syss',
                             'b_kommunenr',
                             'gjeldende_lonn_kr', 
                             'gjeldende_driftsk_kr',
                             'gjeldende_omsetn_kr',
                             'tmp_forbruk_bed',
                             'tmp_salgsint_bed',
                             'tmp_sn2007_5',
                             'n3',
                             'n2',
                             'year']]

    # rename columns
    temp_1 = temp_1.rename(columns={'b_sysselsetting_syss':'syss',
                                    'b_kommunenr':'kommunenr',
                                    'gjeldende_lonn_kr':'lonn',
                                    'gjeldende_omsetn_kr':'oms',
                                    'gjeldende_driftsk_kr': 'drkost',
                                    'tmp_forbruk_bed':'forbruk',
                                    'tmp_salgsint_bed':'salgsint',
                                   })

    temp_1 = temp_1[temp_1['year'] != year]
    
    merged_df["n3"] = merged_df["tmp_sn2007_5"].str[:4]
    merged_df["n2"] = merged_df["tmp_sn2007_5"].str[:2]

    temp_2 = merged_df[['id',
                     'nacef_5',
                     'orgnr_n_1',
                     'b_sysselsetting_syss',
                     'b_kommunenr',
                     'gjeldende_lonn_kr', 
                     'new_drkost',
                     'oms',
                     'bedr_forbruk',
                     'bedr_salgsint',
                     'tmp_sn2007_5',
                     'n3',
                     'n2',
                     'year']]

    temp_2 = temp_2.rename(columns={'b_sysselsetting_syss':'syss',
                                    'b_kommunenr':'kommunenr',
                                    'gjeldende_lonn_kr':'lonn',
                                    'bedr_forbruk':'forbruk',
                                    'bedr_salgsint':'salgsint',
                                    'new_drkost': 'drkost'
                                   })

    temp_2 = temp_2[temp_2['year'] == year]
    
    temp_1['forbruk'] = temp_1['forbruk'].fillna(0)
    temp_1['salgsint'] = temp_1['salgsint'].fillna(0)
    
    timeseries_knn = pd.concat([temp_1, temp_2], axis=0)
    
    # aggregate forbruk per year

    columns_to_convert = ['salgsint', 'forbruk', 'oms', 'drkost', 'lonn', 'syss']

    # Convert columns to integers using pd.to_numeric for safe conversion, errors='coerce' will set issues to NaN
    for column in columns_to_convert:
        timeseries_knn[column] = pd.to_numeric(timeseries_knn[column], errors='coerce')

    timeseries_knn['year'] = timeseries_knn['year'].astype(str)
    timeseries_knn['n3'] = timeseries_knn['n3'].astype(str)

    timeseries_knn['resultat'] = timeseries_knn['oms'] - timeseries_knn['drkost']



    # filter for n3 in 45, 46 or 47
    timeseries_knn = timeseries_knn[timeseries_knn['n2'].isin(['45', '46', '47'])]
    temp = timeseries_knn.copy()
    timeseries_knn_agg = timeseries_knn.groupby(["year", "n3"])[["forbruk", "oms", "drkost", "salgsint", "lonn", 'syss', "resultat"]].sum().reset_index()
    # timeseries_knn_agg = timeseries_knn.groupby(["year", "n3"])[["forbruk", "oms", "drkost", "salgsint", "lonn", 'syss', "resultat"]].sum().reset_index()
    timeseries_knn_agg['lonn_pr_syss'] = timeseries_knn_agg['lonn'] / timeseries_knn_agg['syss']
    timeseries_knn_agg['oms_pr_syss'] = timeseries_knn_agg['oms'] / timeseries_knn_agg['syss']
    timeseries_knn_agg["n2"] = timeseries_knn_agg["n3"].str[:2]



    timeseries_knn__kommune_agg = temp.groupby(["year", "kommunenr", "n3"])[["forbruk", "oms", "drkost", "salgsint", "lonn", 'syss', "resultat"]].sum().reset_index()
    timeseries_knn__kommune_agg['lonn_pr_syss'] = timeseries_knn__kommune_agg['lonn'] / timeseries_knn_agg['syss']
    timeseries_knn__kommune_agg['oms_pr_syss'] = timeseries_knn__kommune_agg['oms'] / timeseries_knn_agg['syss']
    timeseries_knn__kommune_agg["n2"] = timeseries_knn__kommune_agg["n3"].str[:2]
    
    # Create a new column 'n2_f' by extracting the first two characters of 'nacef_5'
    oppdateringsfil['n2_f'] = oppdateringsfil['nacef_5'].str[:2]

    oppdateringsfil['n3_f'] = oppdateringsfil['nacef_5'].str[:4]

    # filter for when 'n2_f' is in 45, 46 or 47

    oppdateringsfil = oppdateringsfil[oppdateringsfil['n2_f'].isin(['45', '46', '47'])]

    # create df called foretak by filtering for when radnr = 1 in oppdateringsfil

    temp = oppdateringsfil[oppdateringsfil['radnr'] == 1]



    # groupby n2_f and sum foretak_omsetning, foretak_driftskostnad, forbruk, salgsint

    temp = temp.groupby('n3_f').sum()[['foretak_omsetning', 'foretak_driftskostnad', 'forbruk', 'salgsint']].reset_index()
    bedrift = oppdateringsfil.groupby('n3_f').sum()[['oms', 'new_drkost', 'bedr_forbruk', 'bedr_salgsint']].reset_index()


    check_totals = temp.merge(bedrift, on='n3_f', how='left')
       
   

    processing_time = time.time() - start_time
    print(f"Time taken to create training data: {processing_time:.2f} seconds")
    
    return oppdateringsfil, timeseries_knn_agg, timeseries_knn__kommune_agg, check_totals, check_manually