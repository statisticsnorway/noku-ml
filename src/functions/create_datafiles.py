import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import pyarrow as pa
import pyarrow.parquet as pq
import gcsfs
import getpass
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
import geopandas as gpd
import sgis as sg
import dapla as dp
import datetime
from dapla.auth import AuthClient
from dapla import FileClient
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import requests
from pyjstat import pyjstat
import sys
sys.path.append("../functions")
import kommune_pop
import kommune_inntekt
import kpi
import ao
fs = FileClient.get_gcs_file_system()
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import multiprocessing
import time
import kommune_translate
import polars as pl
import fsspec


def main(year, limit, skjema_nr, distribtion_percent, tosiffernaring, geo_data=False, uu_data=False):
    
    
    # # Assuming 'skjema' is a variable or a column in a DataFrame
    # if skjema_nr == 'RA-1100':
    #     start_year = 2018
    # else:
    #     start_year = 2017
    prior_year= year - 1
    start_year = 2017

    all_good_dataframes = []  # List to store good dataframes for each year
    all_bad_dataframes = []   # List to store bad dataframes for each year
    all_training_dataframes = []  # List to store training dataframes for each year
    all_time_series_dataframes = []  # List to store time series dataframes for each year
    
    start_datafile_loop = time.time()
    
    for current_year in range(start_year, year + 1):
        
        print("starting data collection for:", {current_year}, "...")
        
        fjor = current_year - 1  # Previous year
        
        # skjema_list = ['RA-0174-1', 'RA-0174A3', 'RA-0827A3']
        # skjema_list = 'RA-0174-1'
        skjema_list = skjema_nr
        
#         fil_path = [
#             f
#             for f in fs.glob(
#                 f"gs://ssb-strukt-naering-data-produkt-prod/naringer/inndata/skjemadata/skjema={skjema_list}/aar={current_year}/*"
#             )
#             # if f.endswith(".parquet")
#         ]

#         # Assuming there's only one file in fil_path
#         if fil_path:
#             skjema = pd.read_parquet(fil_path[0], filesystem=fs)
#         else:
#             raise FileNotFoundError(f"No Parquet files found for year {current_year}")
#             print(fil_path)
               
        felt_id_values = [
            "V_ORGNR",
            "F_ADRESSE",
            "FJOR_NACE_B_T1",
            "TMP_SN2007_5",
            "B_KOMMUNENR",
            "REGTYPE",
            "B_SYSSELSETTING_SYSS",
            "TMP_NY_BDR_SYSS",
            "GJELDENDE_BDR_SYSS",
            "FJOR_SYSSEL_T1",
            "LONN_PST_AORDN",
            "GJELDENDE_LONN_KR",
            "LONN",
            "FJOR_LONN_KR_T1",
            "TMP_SNITTLONN",
            "FJOR_SNITTLONN_T1",
            "GJELDENDE_OMSETN_KR",
            "OMSETN_KR",
            "FJOR_OMSETN_KR_T1",
            "TMP_SNITTOMS",
            "FJOR_SNITTOMS_T1",
            "TMP_SALGSINT_BED",
            "TMP_FORBRUK_BED",
            "VAREKOST_BED",
            "GJELDENDE_DRIFTSK_KR",
            "DRIFTSKOST_KR",
            "FJOR_DRIFTSKOST_KR_T1",
            "NACEF_5",
            "SALGSINT",
            "FORBRUK",
            "TMP_NO_P4005",
            "TMP_AVPROS_ORGFORB",
            "ORGNR_N_1",
            "TMP_NO_OMSETN",
            "TMP_DRIFTSKOSTNAD_9010",
            "TMP_DRIFTSKOSTNAD_9910",
            "TMP_OMS",
            "NO_OMS",
            "B_DRIFTSKOSTNADER",
            "B_OMSETNING",
            "TMP_B_SN07_1",
            "REG_TYPE_BEDRIFT"
        ]

#         # Filter the DataFrame for the specified field values
#         skjema = skjema[skjema["feltnavn"].isin(felt_id_values)]
        
    
        file_path = f"gs://ssb-strukt-naering-data-produkt-prod/naringer/inndata/skjemadata/skjema={skjema_list}/aar={current_year}/skjemadata_data_0.parquet"

        f = FileClient.gcs_open(file_path)
        
        skjema = (
            pl.read_parquet(f)
            .filter(pl.col("feltnavn").is_in(felt_id_values))
        )
        
        skjema = skjema.to_pandas()
        
        skjema.columns = skjema.columns.str.lower()
        
        # Pivot the DataFrame
        skjema = skjema.pivot_table(
            index=["id", "radnr", "lopenr"],
            columns="feltnavn",
            values="feltverdi",
            aggfunc="first",
        )
        skjema = skjema.reset_index()
        skjema.columns = skjema.columns.str.lower()  # Convert column names to lower case
        
        # Assuming 'skjema_list' is a variable, not a column in the DataFrame
        if skjema_list == 'RA-1403':
            # Rename the column 'tmp_oms' to 'tmp_no_omsetn'
            skjema.rename(columns={'tmp_oms': 'tmp_no_omsetn'}, inplace=True)
            
        if skjema_list == 'RA-0255-1':
            # Rename the column 'tmp_oms' to 'tmp_no_omsetn'
            skjema.rename(columns={'no_oms': 'tmp_no_omsetn'}, inplace=True)
            
                        # Get varekostnad data on foretak level
            fil_path = [
                f
                for f in fs.glob(
                    f"gs://ssb-strukt-naering-data-produkt-prod/naringer/klargjorte-data/statistikkfiler/aar={current_year}/statistikkfil_foretak_pub.parquet"
                )
                if f.endswith(".parquet")
            ]

            # Use the ParquetDataset to read multiple files
            # dataset = pq.ParquetDataset(fil_path, filesystem=fs)
            foretak_pub = pd.read_parquet(fil_path, filesystem=fs)
            
#             table = dataset.read()

#             # Convert to Pandas DataFrame
#             foretak_pub = table.to_pandas()

            # Check if current_year is 2022 or higher
            if current_year >= 2023:
                foretak_pub = foretak_pub[['nopost_p4005', 'enhets_id', 'nopost_driftskostnader']]
                foretak_pub.rename(columns={'nopost_driftskostnader': 'tmp_driftskostnad_9010'}, inplace=True)
            else:
                foretak_pub = foretak_pub[['nopost_p4005', 'enhets_id']]


            foretak_pub.rename(columns={'nopost_p4005': 'tmp_no_p4005', 'enhets_id': 'id'}, inplace=True)

            skjema = pd.merge(skjema, foretak_pub, how='left', on='id')
            
            # fill tmp_no_p4005 nan with 0
            skjema['tmp_no_p4005'].fillna(0, inplace=True)
            
            del foretak_pub
            
        if skjema_list == 'RA-1100':
            
            # Get data on foretak level
            fil_path = [
                f
                for f in fs.glob(
                    f"gs://ssb-strukt-naering-data-produkt-prod/naringer/klargjorte-data/statistikkfiler/aar={current_year}/statistikkfil_foretak_pub.parquet"
                )
                if f.endswith(".parquet")
            ]

            # Use the ParquetDataset to read multiple files
            dataset = pq.ParquetDataset(fil_path, filesystem=fs)
            table = dataset.read()

            # Convert to Pandas DataFrame
            foretak_pub = table.to_pandas()

            foretak_pub = foretak_pub[['nopost_p4005', 'enhets_id', 'omsetning', 'naring_f']]

            foretak_pub.rename(columns={'nopost_p4005': 'tmp_no_p4005', 'enhets_id': 'id', 'omsetning': 'tmp_no_omsetn', 'naring_f': 'nacef_5'}, inplace=True)

            skjema = pd.merge(skjema, foretak_pub, how='left', on='id')

            skjema.rename(columns={'b_omsetning': 'gjeldende_omsetn_kr', 'b_driftskostnader': 'driftskost_kr', 'tmp_b_sn07_1': 'tmp_sn2007_5', 'reg_type_bedrift': 'regtype'}, inplace=True)
            
            skjema['gjeldende_bdr_syss'] = skjema['b_sysselsetting_syss']
            
            # fill tmp_no_p4005 nan with 0
            skjema['tmp_no_p4005'].fillna(0, inplace=True)
            
            fil_path = [
                f
                for f in fs.glob(
                    f"gs://ssb-strukt-naering-data-produkt-prod/naringer/klargjorte-data/statistikkfiler/aar={fjor}/statistikkfil_bedrifter_pub.parquet"
                )
                if f.endswith(".parquet")
            ]

            # Use the ParquetDataset to read multiple files
            dataset = pq.ParquetDataset(fil_path, filesystem=fs)
            table = dataset.read()

            # Convert to Pandas DataFrame
            bedrift_pub = table.to_pandas()
            
            bedrift_pub = bedrift_pub[['sysselsetting_syss', 'enhets_id', 'omsetning']]
            
            bedrift_pub.rename(columns={'sysselsetting_syss': 'fjor_syssel_t1', 'enhets_id': 'id', 'omsetning': 'fjor_omsetn_kr_t1'}, inplace=True)
            
            skjema = pd.merge(skjema, bedrift_pub, how='left', on='id')
            
            skjema['fjor_syssel_t1'].fillna(0, inplace=True)
        
            
            del bedrift_pub, dataset, table        
        
        # Foretak level data is always when radnr = 0
        foretak = skjema.loc[skjema["radnr"] == 0]

        # Create the 'bedrift' DataFrame
        bedrift = skjema.loc[skjema["radnr"] > 0]

        if current_year >= 2022:
            selected_columns = [
                "id",
                "lopenr",
                "forbruk",
                "orgnr_n_1",
                "nacef_5",
                "salgsint",
                "tmp_driftskostnad_9010",
                "tmp_no_omsetn",
                "tmp_no_p4005",
            ]
        else:
            selected_columns = [
                "id",
                "lopenr",
                "forbruk",
                "nacef_5",
                "orgnr_n_1",
                "salgsint",
                "tmp_driftskostnad_9010",
                # "tmp_driftskostnad_9910",
                "tmp_no_omsetn",
                "tmp_no_p4005",
            ]

        foretak = foretak[selected_columns]
        
        nacef_prefix_list = foretak['nacef_5'].str[:2].unique().tolist()

        # Display the list
        print(nacef_prefix_list)

        # Assuming 'foretak' is your DataFrame
        foretak.rename(columns={"tmp_no_omsetn": "foretak_omsetning"}, inplace=True)


        foretak = foretak.fillna(0)
        
        # Apply numeric conversion and find max for tmp_driftskostnad_9010 only
        foretak["tmp_driftskostnad_9010"] = pd.to_numeric(foretak["tmp_driftskostnad_9010"], errors="coerce")
        foretak["foretak_driftskostnad"] = foretak["tmp_driftskostnad_9010"]

        # Drop the specified column
        foretak.drop(["tmp_driftskostnad_9010"], axis=1, inplace=True)

        columns_to_drop = [
            "forbruk",
            "nacef_5",
            "orgnr_n_1",
            "salgsint",
            "tmp_no_omsetn",
            "tmp_no_p4005",
        ]

#         if current_year >= 2022:
#             # Apply numeric conversion and find max for tmp_driftskostnad_9010 only
#             foretak["tmp_driftskostnad_9010"] = pd.to_numeric(foretak["tmp_driftskostnad_9010"], errors="coerce")
#             foretak["foretak_driftskostnad"] = foretak["tmp_driftskostnad_9010"]

#             # Drop the specified column
#             foretak.drop(["tmp_driftskostnad_9010"], axis=1, inplace=True)

#             columns_to_drop = [
#                 "forbruk",
#                 "nacef_5",
#                 "orgnr_n_1",
#                 "salgsint",
#                 "tmp_no_omsetn",
#                 "tmp_no_p4005",
#             ]
#         else:
#             # Apply numeric conversion and find max for both columns
#             foretak[["tmp_driftskostnad_9010", "tmp_driftskostnad_9910"]] = foretak[
#                 ["tmp_driftskostnad_9010", "tmp_driftskostnad_9910"]
#             ].apply(pd.to_numeric, errors="coerce")

#             foretak["foretak_driftskostnad"] = foretak[
#                 ["tmp_driftskostnad_9010", "tmp_driftskostnad_9910"]
#             ].max(axis=1)

#             # Drop the specified columns
#             foretak.drop(["tmp_driftskostnad_9010", "tmp_driftskostnad_9910"], axis=1, inplace=True)

#             columns_to_drop = [
#                 "forbruk",
#                 "nacef_5",
#                 "orgnr_n_1",
#                 "salgsint",
#                 "tmp_no_omsetn",
#                 "tmp_no_p4005",
#             ]


        bedrift.drop(columns_to_drop, axis=1, inplace=True)
    
        # Replace commas with dots in the specified columns

        columns_to_fill = ["gjeldende_omsetn_kr", "driftskost_kr"]
        
        bedrift[columns_to_fill] = bedrift[columns_to_fill].replace(',', '.', regex=True)

        # Convert columns to numeric, replacing non-convertible values with NaN
        bedrift[columns_to_fill] = bedrift[columns_to_fill].apply(
            pd.to_numeric, errors="coerce"
        )
        
        # if gjeldende_omsetn_kr or driftskost_kr is negative then change to = 0
        bedrift[columns_to_fill] = bedrift[columns_to_fill].applymap(lambda x: x if x > 0 else 0)

        # Fill NaN values with 0 for the specified columns
        bedrift[columns_to_fill] = bedrift[columns_to_fill].fillna(0)
        

        # Group by 'id' and calculate the sum
        grouped_bedrift = (
            bedrift.groupby("id")[["gjeldende_omsetn_kr", "driftskost_kr"]].sum().reset_index()
        )

        # Rename the columns
        grouped_bedrift.rename(
            columns={"gjeldende_omsetn_kr": "tot_oms_fordelt", "driftskost_kr": "tot_driftskost_fordelt"},
            inplace=True,
        )

        # Merge the grouped DataFrame back to the original DataFrame based on 'id'
        bedrift = pd.merge(bedrift, grouped_bedrift, on="id", how="left")

        merged_df = pd.merge(foretak, bedrift, on=["id", "lopenr"], how="inner")
        
        
        # Convert columns to numeric, replacing non-convertible values with NaN
        merged_df["tot_oms_fordelt"] = pd.to_numeric(
            merged_df["tot_oms_fordelt"], errors="coerce"
        )
        merged_df["foretak_omsetning"] = pd.to_numeric(
            merged_df["foretak_omsetning"], errors="coerce"
        )

        # Calculate omsetning_percentage
        merged_df["omsetning_percentage"] = (
            merged_df["tot_oms_fordelt"] / merged_df["foretak_omsetning"]
        )

        # Convert columns to numeric, replacing non-convertible values with NaN
        merged_df["tot_driftskost_fordelt"] = pd.to_numeric(
            merged_df["tot_driftskost_fordelt"], errors="coerce"
        )
        merged_df["foretak_driftskostnad"] = pd.to_numeric(
            merged_df["foretak_driftskostnad"], errors="coerce"
        )

        # Calculate driftskostnader_percentage
        merged_df["driftskostnader_percentage"] = (
            merged_df["tot_driftskost_fordelt"] / merged_df["foretak_driftskostnad"]
        )

        merged_df["driftskostnader_percentage"] = (
            merged_df["tot_driftskost_fordelt"] / merged_df["foretak_driftskostnad"]
        ).round(4)

        # Fill NaN with a specific value (e.g., 0)
        merged_df["driftskostnader_percentage"].fillna(0, inplace=True)
        merged_df["omsetning_percentage"].fillna(0, inplace=True)
        
        # Create the 'Good' DataFrame
        good_temp_df = merged_df[
            (merged_df["omsetning_percentage"] >= limit)
            & (merged_df["driftskostnader_percentage"] >= limit)
        ]

        # Create 'bedrift_count' and 'distribution_count'
        good_temp_df["bedrift_count"] = good_temp_df.groupby("orgnr_n_1")[
            "orgnr_n_1"
        ].transform("count")
        good_temp_df["distribution_count"] = good_temp_df.groupby("orgnr_n_1")[
            "gjeldende_omsetn_kr"
        ].transform(lambda x: (x > 0).sum())
        
        # Calculate 'bedrift_count' where 'gjeldende_bdr_syss' is not equal to 0
        # Calculate 'bedrift_count' per 'orgnr_n_1'
#         bedrift_counts = good_temp_df.groupby('orgnr_n_1')['gjeldende_bdr_syss'].apply(lambda x: (x != 0).sum())

#         # Map the counts back to the DataFrame
#         good_temp_df['bedrift_count'] = good_temp_df['orgnr_n_1'].map(bedrift_counts)

#         # Calculate 'distribution_count' per 'orgnr_n_1'
#         distribution_counts = good_temp_df.groupby('orgnr_n_1').apply(
#             lambda g: ((g['gjeldende_bdr_syss'] != 0) & (g['gjeldende_omsetn_kr'] > 0)).sum()
#         )

#         # Map the counts back to the DataFrame
#         good_temp_df['distribution_count'] = good_temp_df['orgnr_n_1'].map(distribution_counts)



        # Create 'bad_temp' DataFrame based on conditions
        # bad_temp = good_temp_df[
        #     (good_temp_df["bedrift_count"] >= 2) & (good_temp_df["distribution_count"] < 2)
        # ]
        
        good_temp_df['distribution_rate'] = good_temp_df['distribution_count'] / good_temp_df['bedrift_count']
        
        bad_temp = good_temp_df[
            (good_temp_df["bedrift_count"] >= 2) & (good_temp_df["distribution_rate"] <= distribtion_percent)
        ]                
        
        bad_temp['driftskost_kr'] = np.nan

        # Create 'good_df' by excluding rows from 'bad_temp'
        good_df = (
            pd.merge(good_temp_df, bad_temp, how="outer", indicator=True)
            .query('_merge == "left_only"')
            .drop("_merge", axis=1)
        )

        # Create the 'Mixed' DataFrame
        onlygoodoms = merged_df[
            (
                (merged_df["omsetning_percentage"] > limit)
                & (merged_df["driftskostnader_percentage"] <= limit)
            )
        ]
        
        onlygoodoms['driftskost_kr'] = np.nan

        onlygooddriftskostnader = merged_df[
            (
                (merged_df["driftskostnader_percentage"] > limit)
                & (merged_df["omsetning_percentage"] <= limit)
            )
        ]

        # Create the 'Bad' DataFrame
        bad_df = merged_df[
            (merged_df["omsetning_percentage"] <= limit)
            & (merged_df["driftskostnader_percentage"] <= limit)
        ]
        bad_df = pd.concat([bad_df, bad_temp]).drop_duplicates(keep=False)
        bad_df = pd.concat([bad_df, onlygooddriftskostnader]).drop_duplicates(keep=False)
        
        good_df = pd.concat([good_df, onlygoodoms]).drop_duplicates(keep=False)
        
        good_df["oms_share"] = good_df["gjeldende_omsetn_kr"] / good_df["tot_oms_fordelt"].round(5)

        # Round the values to whole numbers before assigning to the new columns
        good_df["new_oms"] = (
            (good_df["oms_share"] * good_df["foretak_omsetning"]).round(0).astype(int)
        )
        
        # swapped order of this concat , because we want even the bad data to remain in order to create trend lines. 
        bad_df["new_oms"] = bad_df["gjeldende_omsetn_kr"]
        merged_df = pd.concat([good_df, bad_df], ignore_index=True)
        
        # merged_df = good_df.copy()
        
        time_series_df = merged_df.copy()

        # bad_df["new_oms"] = bad_df["gjeldende_omsetn_kr"]

        del onlygooddriftskostnader
                 
        if uu_data and (current_year == year or current_year == prior_year):
            
            start_uu = time.time()
            
            print("uu_data for:", {current_year}, "is True, proceeding with data processing...")
            
            temp_prior_year= current_year - 1
            
#             fil_path = [
#                 f
#                 for f in fs.glob(
#                     f"gs://ssb-strukt-naering-data-produkt-prod/naringer/klargjorte-data/statistikkfiler/aar={current_year}/statistikkfil_bedrifter_pub.parquet"
#                 )
#                 if f.endswith(".parquet")
#             ]
            
        

#             # Use the ParquetDataset to read multiple files
#             dataset = pq.ParquetDataset(fil_path, filesystem=fs)
#             table = dataset.read()

#             # Convert to Pandas DataFrame
#             bedrift_pub = table.to_pandas()

            columns_needed = [
                'ts_forbruk', 'naring_f', 'orgnr_foretak', 'ts_salgsint', 'omsetning',
                'nopost_p4005', 'nopost_driftskostnader', 'kommune', 'sysselsetting_syss',
                'naring', 'orgnr_bedrift', 'reg_type_f', 'type'
            ]

            file_path = f"gs://ssb-strukt-naering-data-produkt-prod/naringer/klargjorte-data/statistikkfiler/aar={current_year}/statistikkfil_bedrifter_pub.parquet"

            f = FileClient.gcs_open(file_path)
            bedrift_pub = pl.read_parquet(f, columns=columns_needed)

            bedrift_pub = bedrift_pub.to_pandas()
            
            bedrift_pub.columns = bedrift_pub.columns.str.lower()

            # filter for when reg_type_f = 01
            bedrift_pub = bedrift_pub[bedrift_pub['reg_type_f'] == '01']
            bedrift_pub = bedrift_pub[bedrift_pub['type'] != 'S']

            bedrift_pub = bedrift_pub[['ts_forbruk', 'naring_f', 'orgnr_foretak', 'ts_salgsint', 'omsetning', 'nopost_p4005', 'nopost_driftskostnader', 'kommune', 'sysselsetting_syss', 'naring', 'orgnr_bedrift']]

            bedrift_pub['lopenr'] = 1
            bedrift_pub['radnr'] = 1

            bedrift_pub['driftskostnader_percentage'] = 1
            bedrift_pub['omsetning_percentage'] = 1

            # rename variables

            bedrift_pub.rename(columns={'ts_forbruk': 'forbruk', 'naring_f': 'nacef_5', 'orgnr_foretak': 'orgnr_n_1', 'ts_salgsint': 'salgsint', 'omsetning': 'foretak_omsetning', 'nopost_p4005': 'tmp_no_p4005', 'nopost_driftskostnader': 'foretak_driftskostnad', 'kommune': 'b_kommunenr', 'sysselsetting_syss': 'b_sysselsetting_syss', 'reg_type_f': 'regtype', 'naring': 'tmp_sn2007_5', 'orgnr_bedrift': 'v_orgnr'}, inplace=True)

            bedrift_pub['gjeldende_omsetn_kr'] = bedrift_pub['foretak_omsetning']
            bedrift_pub['omsetn_kr'] = bedrift_pub['foretak_omsetning']
            bedrift_pub['new_oms'] = bedrift_pub['foretak_omsetning']
            bedrift_pub['tot_oms_fordelt'] = bedrift_pub['foretak_omsetning']
            bedrift_pub['driftskost_kr'] = bedrift_pub['foretak_driftskostnad']
            bedrift_pub['gjeldende_driftsk_kr'] = bedrift_pub['foretak_driftskostnad']
            bedrift_pub['tot_driftskost_fordelt'] = bedrift_pub['foretak_driftskostnad']
            bedrift_pub['gjeldende_bdr_syss'] = bedrift_pub['b_sysselsetting_syss']
            bedrift_pub['tmp_forbruk_bed'] = bedrift_pub['forbruk']
            bedrift_pub['tmp_salgsint_bed'] = bedrift_pub['salgsint']
            bedrift_pub['id'] = bedrift_pub['orgnr_n_1']
            
#             fil_path = [
#                 f
#                 for f in fs.glob(
#                     f"gs://ssb-strukt-naering-data-produkt-prod/naringer/klargjorte-data/statistikkfiler/aar={temp_prior_year}/statistikkfil_bedrifter_pub.parquet"
#                 )
#                 if f.endswith(".parquet")
#             ]

#             # Use the ParquetDataset to read multiple files
#             dataset = pq.ParquetDataset(fil_path, filesystem=fs)
#             table = dataset.read()

#             # Convert to Pandas DataFrame
#             bedrift_pub_x = table.to_pandas()

            columns_needed_x = [
                'reg_type_f', 'orgnr_bedrift', 'sysselsetting_syss'
            ]

            file_path = f"gs://ssb-strukt-naering-data-produkt-prod/naringer/klargjorte-data/statistikkfiler/aar={temp_prior_year}/statistikkfil_bedrifter_pub.parquet"

            f = FileClient.gcs_open(file_path)
            bedrift_pub_x = pl.read_parquet(f, columns=columns_needed_x)

            bedrift_pub_x = bedrift_pub_x.to_pandas()
            
            bedrift_pub_x.columns = bedrift_pub_x.columns.str.lower()

            bedrift_pub_x = bedrift_pub_x[bedrift_pub_x['reg_type_f'] == '01']


            bedrift_pub_x = bedrift_pub_x[['orgnr_bedrift', 'sysselsetting_syss']]

            bedrift_pub_x.rename(columns={'sysselsetting_syss': 'fjor_syssel_t1', 'orgnr_bedrift': 'v_orgnr'}, inplace=True)
            
            bedrift_pub = pd.merge(bedrift_pub, bedrift_pub_x, how='left', on='v_orgnr')


            # fill nan for fjor_syssel_t1 with 0

            bedrift_pub['fjor_syssel_t1'] = bedrift_pub['fjor_syssel_t1'].fillna(0)
            
            bedrift_pub['temp_n2'] = bedrift_pub['nacef_5'].str[:2]
            
            print("uu data shape before filtering",bedrift_pub.shape)
            
            bedrift_pub = bedrift_pub[bedrift_pub['temp_n2'].isin(nacef_prefix_list)]
            
            bedrift_pub = bedrift_pub.drop(columns=['temp_n2'])
            
            print("uu data shape after filtering",bedrift_pub.shape)

            del bedrift_pub_x

            merged_df = pd.concat([merged_df, bedrift_pub])
            

            del bedrift_pub
            
            # Calculate processing time
            processing_time_uu = time.time() - start_uu
            print(f"Time taken to process uu data for {current_year}: {processing_time_uu:.2f} seconds")
        
        else:
            print("uu_data is False, skipping data processing.")
        
        merged_df["n4"] = merged_df["nacef_5"].str[:5]


        # kommune_befolk = kommune_pop.befolkning_behandling(current_year, fjor)
        # kommune_inn = kommune_inntekt.inntekt_behandling(current_year, fjor)
        # kpi_df = kpi.process_kpi_data(current_year)
        
        # Get kommune population growth , income trends and inflation data 
        
        api_time = time.time()
        
        try:
            kommune_befolk = kommune_pop.befolkning_behandling(current_year, fjor)
        except Exception as e:
            print(f"Failed to fetch kommune_befolk for {current_year}. Trying for {current_year - 1}.")
            try:
                kommune_befolk = kommune_pop.befolkning_behandling(current_year - 1, fjor - 1)
            except Exception as e:
                print(f"Failed to fetch kommune_befolk for {current_year - 1} as well.")
                kommune_befolk = None

        try:
            kommune_inn = kommune_inntekt.inntekt_behandling(current_year, fjor)
        except Exception as e:
            print(f"Failed to fetch kommune_inn for {current_year}. Trying for {current_year - 1}.")
            try:
                kommune_inn = kommune_inntekt.inntekt_behandling(current_year - 1, fjor - 1)
            except Exception as e:
                print(f"Failed to fetch kommune_inn for {current_year - 1} as well.")
                kommune_inn = None

        try:
            kpi_df = kpi.process_kpi_data(current_year)
        except Exception as e:
            print(f"Failed to fetch kpi_df for {current_year}. Trying for {current_year - 1}.")
            try:
                kpi_df = kpi.process_kpi_data(current_year - 1)
            except Exception as e:
                print(f"Failed to fetch kpi_df for {current_year - 1} as well.")
                kpi_df = None
                
        processing_time_api = time.time() - api_time
        print(f"Time taken to process kommune data, population data and inflation data for {current_year}: {processing_time_api:.2f} seconds")
        
        
        # Convert string columns to numeric
        merged_df["gjeldende_bdr_syss"] = pd.to_numeric(
            merged_df["gjeldende_bdr_syss"], errors="coerce"
        )
        merged_df["fjor_syssel_t1"] = pd.to_numeric(
            merged_df["fjor_syssel_t1"], errors="coerce"
        )

        # Perform division after conversion
        merged_df["emp_delta"] = merged_df["gjeldende_bdr_syss"] / merged_df["fjor_syssel_t1"]

        imputable_df = merged_df.copy()
        
        test3 = imputable_df.copy()


        imputable_df = imputable_df.drop_duplicates(subset=["v_orgnr"])

        # imputable_df['n4'] =  imputable_df['nacef_5'].str[:5]
        imputable_df["n4"] = imputable_df["tmp_sn2007_5"].str[:5]

        imputable_df = pd.merge(imputable_df, kommune_befolk, on=["b_kommunenr"], how="left")
        imputable_df = pd.merge(imputable_df, kommune_inn, on=["b_kommunenr"], how="left")
        imputable_df = pd.merge(imputable_df, kpi_df, on=["n4"], how="left")

        # Ensure columns are numeric
        imputable_df["fjor_omsetn_kr_t1"] = pd.to_numeric(
            imputable_df["fjor_omsetn_kr_t1"], errors="coerce"
        )
        imputable_df["inflation_rate"] = pd.to_numeric(
            imputable_df["inflation_rate"], errors="coerce"
        )
        imputable_df["befolkning_delta"] = pd.to_numeric(
            imputable_df["befolkning_delta"], errors="coerce"
        )
        imputable_df["emp_delta"] = pd.to_numeric(imputable_df["emp_delta"], errors="coerce")
        imputable_df["inntekt_delta"] = pd.to_numeric(
            imputable_df["inntekt_delta"], errors="coerce"
        )

        # general_inflation_rate = imputable_df.loc[
        #     imputable_df["n4"] == "47.78", "inflation_rate"
        # ].values[0]
        
        
        # Fetch the general inflation rate for the current year
        general_inflation_rate = kpi.fetch_general_inflation_rate(current_year)

        # If fetching for the current year fails, try fetching for the previous year
        # if general_inflation_rate is None:
        #     general_inflation_rate = kpi.fetch_general_inflation_rate(current_year - 1)

        # Fill missing inflation rate values with the fetched general inflation rate
        imputable_df["inflation_rate"] = imputable_df["inflation_rate"].fillna(general_inflation_rate)

        
        # imputable_df["inflation_rate"] = imputable_df["inflation_rate"].fillna(
        #     general_inflation_rate
        # )

        imputable_df["inflation_rate_oms"] = (
            imputable_df["fjor_omsetn_kr_t1"] * imputable_df["inflation_rate"]
        )
        imputable_df["befolkning_delta_oms"] = (
            imputable_df["fjor_omsetn_kr_t1"] * imputable_df["befolkning_delta"]
        )
        imputable_df["emp_delta_oms"] = (
            imputable_df["fjor_omsetn_kr_t1"] * imputable_df["emp_delta"]
        )
        imputable_df["inntekt_delta_oms"] = (
            imputable_df["fjor_omsetn_kr_t1"] * imputable_df["inntekt_delta"]
        )

        # Treat Nan for inflation_rate_oms
        imputable_df["inflation_rate_oms"].replace([np.inf, -np.inf], np.nan, inplace=True)
        group_means = imputable_df.groupby("nacef_5")["inflation_rate_oms"].transform("mean")
        # Step 3: Fill NaN values in 'inflation_rate_oms' with the corresponding group's mean
        imputable_df["inflation_rate_oms"].fillna(group_means, inplace=True)


        categories_to_impute = [
            "emp_delta_oms",
            "befolkning_delta_oms",
            "inntekt_delta_oms",
            "inflation_rate_oms",
        ]

        # Identify rows where 'b_sysselsetting_syss' is equal to 0
        rows_to_impute = imputable_df["b_sysselsetting_syss"] == 0

        # Replace NaN values with 0 for the identified rows and specified categories
        imputable_df.loc[rows_to_impute, categories_to_impute] = imputable_df.loc[
            rows_to_impute, categories_to_impute
        ].fillna(0)


        # Group by 'tmp_sn2007_5' and calculate the average 'emp_delta_oms'
        average_foretak_oms_pr_naring = imputable_df.groupby("tmp_sn2007_5")[
            "foretak_omsetning"
        ].mean()

        # Create a new column 'average_emp_delt_oms_pr_naring' and assign the calculated averages to it
        imputable_df["average_emp_delt_oms_pr_naring"] = imputable_df["nacef_5"].map(
            average_foretak_oms_pr_naring
        )

        # Fill NaN values with 0 before rounding and converting to int
        imputable_df["average_emp_delt_oms_pr_naring"] = imputable_df["average_emp_delt_oms_pr_naring"].fillna(0).round(0).astype(int)


        imputable_df["average_emp_delt_oms_pr_naring"] = (
            imputable_df["average_emp_delt_oms_pr_naring"].round(0).astype(int)
        )

        knn_df = imputable_df[
            [
                "average_emp_delt_oms_pr_naring",
                "emp_delta_oms",
                "befolkning_delta_oms",
                "inflation_rate_oms",
                "inntekt_delta_oms",
                "b_sysselsetting_syss",
                "v_orgnr",
            ]
        ]
        knn_df = knn_df.replace([np.inf, -np.inf], np.nan)


        imputable_df_copy = knn_df.copy()

        # Define the columns for numerical features
        numerical_features = [
            "average_emp_delt_oms_pr_naring",
            "emp_delta_oms",
            "befolkning_delta_oms",
            "inflation_rate_oms",
            "inntekt_delta_oms",
            "b_sysselsetting_syss",
        ]

        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[("num", StandardScaler(), numerical_features)],
            remainder="passthrough",  # Keep non-numerical columns unchanged
        )

        knn_df["v_orgnr"] = knn_df["v_orgnr"].astype(str)

        # Ensure all columns in knn_df are numeric except 'v_orgnr'
        knn_df[numerical_features] = knn_df[numerical_features].apply(pd.to_numeric, errors="coerce")

        # Create KNN imputer
        knn_imputer = KNNImputer(n_neighbors=3)

        # Create imputer pipeline
        imputer_pipeline = Pipeline([("preprocessor", preprocessor), ("imputer", knn_imputer)])

        # Fit and transform the copy of your DataFrame
        imputed_values = imputer_pipeline.fit_transform(knn_df[numerical_features])

        # Convert the imputed values back to a DataFrame and merge 'v_orgnr' column
        imputed_knn_df = pd.DataFrame(imputed_values, columns=numerical_features)
        imputed_knn_df["v_orgnr"] = knn_df["v_orgnr"].values

        # Inverse transform the scaled numerical features
        inverse_scaled_features = preprocessor.named_transformers_["num"].inverse_transform(
            imputed_knn_df[numerical_features].values
        )
        imputed_knn_df[numerical_features] = inverse_scaled_features

        knn_df = imputed_knn_df.copy()

        # knn_df["v_orgnr"] = knn_df["v_orgnr"].round(0).astype(int)
        knn_df["v_orgnr"] = knn_df["v_orgnr"].astype(object)
        columns_to_drop = ["average_emp_delt_oms_pr_naring", "b_sysselsetting_syss"]
        knn_df.drop(columns=columns_to_drop, axis=1, inplace=True)


        columns_to_drop = [
            "emp_delta_oms",
            "befolkning_delta_oms",
            "inflation_rate_oms",
            "inntekt_delta_oms",
        ]

        imputable_df.drop(columns=columns_to_drop, axis=1, inplace=True)


        knn_df["v_orgnr"] = knn_df["v_orgnr"].astype(str)
        imputable_df["v_orgnr"] = imputable_df["v_orgnr"].astype(str)

        # Strip 'v_orgnr' column in both knn_df and imputable_df
        knn_df["v_orgnr"] = knn_df["v_orgnr"].str.strip()
        imputable_df["v_orgnr"] = imputable_df["v_orgnr"].str.strip()

        imputable_df = pd.merge(imputable_df, knn_df, how="inner", on="v_orgnr")

        
        # Leave on or off?
        imputable_df_filtered = imputable_df[~imputable_df["regtype"].isin(["04", "11"])]


        columns_for_imputation = [
            "new_oms",
            "nacef_5",
            "inntekt_delta_oms",
            "emp_delta_oms",
            "befolkning_delta_oms",
            "inflation_rate_oms",
            "v_orgnr",
            "gjeldende_bdr_syss",
        ]

        filtered_imputation_df = imputable_df_filtered[columns_for_imputation]
        filtered_imputation_df.replace([np.inf, -np.inf], np.nan, inplace=True)


        columns_to_drop = [
            "emp_delta_oms",
            "befolkning_delta_oms",
            "inflation_rate_oms",
            "inntekt_delta_oms",
        ]

        imputable_df.drop(columns=columns_to_drop, axis=1, inplace=True)

        columns_for_imputation = filtered_imputation_df.columns.tolist()
        columns_for_imputation.remove("new_oms")

        # Filter for rows where all columns (except 'new_oms') have no NaN values
        cleaned_imputation_df = filtered_imputation_df.dropna(
            subset=columns_for_imputation, how="any"
        )

        # Filter for rows where at least one column (excluding 'new_oms') has NaN values
        nn_df = filtered_imputation_df[
            filtered_imputation_df[columns_for_imputation].isna().any(axis=1)
        ]

        cleaned_imputation_df["inflation_rate_oms"] = (
            cleaned_imputation_df["inflation_rate_oms"].round(0).astype(int)
        )
        cleaned_imputation_df["befolkning_delta_oms"] = (
            cleaned_imputation_df["befolkning_delta_oms"].round(0).astype(int)
        )
        cleaned_imputation_df["emp_delta_oms"] = (
            cleaned_imputation_df["emp_delta_oms"].round(0).astype(int)
        )
        cleaned_imputation_df["inntekt_delta_oms"] = (
            cleaned_imputation_df["inntekt_delta_oms"].round(0).astype(int)
        )

        filtered_indices = filtered_imputation_df.index.tolist()

        # Filter rows from filtered_imputation_df that are not present in cleaned_imputation_df (based on index)
        nn_df = filtered_imputation_df[
            ~filtered_imputation_df.index.isin(cleaned_imputation_df.index)
        ]

        training_data = pd.merge(
            cleaned_imputation_df,
            imputable_df[["v_orgnr", "tmp_sn2007_5", "b_kommunenr", "b_sysselsetting_syss", 'id', 'orgnr_n_1']],
            how="left",
            on=["v_orgnr"]
        )

        training_data["b_sysselsetting_syss"] = training_data["b_sysselsetting_syss"].fillna(0)

        training_data["b_sysselsetting_syss"] = pd.to_numeric(
            training_data["b_sysselsetting_syss"], errors="coerce"
        )

        # Now you can round the values and convert them to integers
        training_data["b_sysselsetting_syss"] = (
            training_data["b_sysselsetting_syss"].round(0).astype("Int64"))
        
        training_data['year'] = current_year
        good_df['year'] = current_year
        bad_df['year'] = current_year
        time_series_df['year'] = current_year
        
       
        # Create the DataFrames
        
        all_good_dataframes.append(good_df)
        all_bad_dataframes.append(bad_df)
        all_training_dataframes.append(training_data)
        all_time_series_dataframes.append(time_series_df)
        
        print("finished collecting and cleaning data for year:", current_year)
        
    # Concatenate all DataFrames into a single DataFrame
    training_data = pd.concat(all_training_dataframes, ignore_index=True)
    bad_data = pd.concat(all_bad_dataframes, ignore_index=True)
    good_data = pd.concat(all_good_dataframes, ignore_index=True)
    time_series_df = pd.concat(all_time_series_dataframes, ignore_index=True)

    current_year_good_oms = good_data[good_data['year'] == year]
    current_year_bad_oms = bad_data[bad_data['year'] == year]
    v_orgnr_list_for_imputering = current_year_bad_oms['v_orgnr'].tolist()
    # unique_id_list = current_year_bad_oms[current_year_bad_oms['nacef_5'].str.startswith(tosiffernaring)]['id'].unique().tolist()   
    # If tosiffernaring contains multiple categories, filter by checking if 'nacef_5' starts with any of them
    
    unique_id_list = current_year_bad_oms[
        current_year_bad_oms['nacef_5'].str[:2].isin(tosiffernaring)
    ]['id'].unique().tolist()

    
    # Easy solution for filling Nan Values - only for training, not for editing real data
    training_data['tmp_sn2007_5'].fillna(training_data['nacef_5'], inplace=True)
    training_data['b_kommunenr'].fillna('0301', inplace=True)
    
    # Replace commas with periods in the 'new_oms' column and convert to float
    # training_data['new_oms'] = training_data['new_oms'].str.replace(',', '.').astype(float)
    
    # Create trend data
    
    # Calculate processing time
    processing_time_datafile_loop = time.time() - start_datafile_loop
    print(f"Time taken to create base training data: {processing_time_datafile_loop:.2f} seconds")
    
    oms_trend_time = time.time()
    
    print("starting regression line function")
    # Determine the number of CPU cores available
    num_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores available: {num_cores}")

    numerical_columns = [
        "new_oms"
    ]

    # Sort the data
    training_data = training_data.sort_values(by=["v_orgnr", "year"])
    
    # Function to process each group
    def process_group(v_orgnr, group):
        group_forecast = group[["v_orgnr", "year"]].copy()
        for col in numerical_columns:
            if col == "new_oms" and group[col].isna().any():
                group_forecast[f"{col}_trendForecast"] = np.nan
                continue
            X = group[group["year"] < year][["year"]]
            y = group[group["year"] < year][col]
            if len(X) > 1:
                model = LinearRegression()
                model.fit(X, y)
                current_year = pd.DataFrame({"year": [year]})
                forecast = model.predict(current_year)[0]
                group_forecast[f"{col}_trendForecast"] = model.predict(group[["year"]])
                group_forecast.loc[group_forecast["year"] == year, f"{col}_trendForecast"] = forecast
            else:
                group_forecast[f"{col}_trendForecast"] = np.nan
        return group_forecast

    # Parallel processing
    results = Parallel(n_jobs=num_cores)(
        delayed(process_group)(v_orgnr, group)
        for v_orgnr, group in training_data.groupby("v_orgnr")
    )

    # Concatenate results
    trend_forecasts = pd.concat(results, ignore_index=True)
    
    processing_time_trends = time.time() - start_datafile_loop
    print(f"Time taken to create base training data: {processing_time_trends:.2f} seconds")

    # Merge the trend forecasts with the original training data
    training_data = pd.merge(training_data, trend_forecasts, on=["v_orgnr", "year"], how="left")
    
    # fill nan with 0 
    
    # training_data.select_dtypes(include=['number']).fillna(0, inplace=True)
    
    
    # Ensure 'new_oms' and 'gjeldende_bdr_syss' are numeric
    training_data['new_oms'] = pd.to_numeric(training_data['new_oms'], errors='coerce')
    
    # Fill NaN values in 'new_oms_trendForecast' with values from 'new_oms'
    training_data['new_oms_trendForecast'].fillna(training_data['new_oms'], inplace=True)
    
    # Count the number of NaN values in each column
    nan_counts = training_data.isna().sum()

    # Print the result
    print("Number of NaN values in training variables")
    print(nan_counts)

    training_data['gjeldende_bdr_syss'] = pd.to_numeric(training_data['gjeldende_bdr_syss'], errors='coerce')
    
    # remove temp orgnr rows. 
    training_data = training_data[~training_data['v_orgnr'].isin(['111111111', '123456789'])]
    
    training_data = kommune_translate.translate_kommune_kodes_2(training_data)
    
    avg_new_oms_per_tmp_sn2007_5 = training_data.groupby(['tmp_sn2007_5', 'year']).apply(
        lambda x: (x['new_oms'] / x['gjeldende_bdr_syss']).replace([np.inf, -np.inf], np.nan).mean()).reset_index()
    avg_new_oms_per_tmp_sn2007_5.columns = ['tmp_sn2007_5', 'year', 'avg_new_oms_per_gjeldende_bdr_syss']


    # Calculate the average new_oms per gjeldende_bdr_syss for each tmp_sn2007_5, b_kommunenr, and year
    avg_new_oms_per_tmp_sn2007_5_per_b_kommunenr = training_data.groupby(['tmp_sn2007_5', 'b_kommunenr', 'year']).apply(
        lambda x: (x['new_oms'] / x['gjeldende_bdr_syss']).replace([np.inf, -np.inf], np.nan).mean()).reset_index()
    avg_new_oms_per_tmp_sn2007_5_per_b_kommunenr.columns = ['tmp_sn2007_5', 'b_kommunenr', 'year', 'avg_new_oms_per_gjeldende_bdr_syss_kommunenr']
    
    training_data = pd.merge(training_data, avg_new_oms_per_tmp_sn2007_5, on=['tmp_sn2007_5', 'year'], how='left')
    training_data = pd.merge(training_data, avg_new_oms_per_tmp_sn2007_5_per_b_kommunenr, on=['tmp_sn2007_5', 'b_kommunenr', 'year'], how='left')
    
    training_data['oms_syssmean_basedOn_naring'] = training_data['avg_new_oms_per_gjeldende_bdr_syss'] * training_data['gjeldende_bdr_syss']
    training_data['oms_syssmean_basedOn_naring_kommune'] = training_data['avg_new_oms_per_gjeldende_bdr_syss_kommunenr'] * training_data['gjeldende_bdr_syss']
    
    # Fill NaN values in specified columns with 0 if 'gjeldende_bdr_syss' is 0
    columns_to_fill = [
        'avg_new_oms_per_gjeldende_bdr_syss',
        'avg_new_oms_per_gjeldende_bdr_syss_kommunenr',
        'oms_syssmean_basedOn_naring',
        'oms_syssmean_basedOn_naring_kommune'
    ]

    # Apply the condition and fill NaN values
    for col in columns_to_fill:
        training_data.loc[training_data['gjeldende_bdr_syss'] == 0, col] = training_data.loc[training_data['gjeldende_bdr_syss'] == 0, col].fillna(0)

    # Add geographical data:
    
    def geo(training_data):
        current_date = datetime.datetime.now()

        # Format the year and month
        current_year = current_date.strftime("%Y")
        current_year_int = int(current_date.strftime("%Y"))
        current_month = current_date.strftime("%m")

        # Subtract one day from the first day of the current month to get the last day of the previous month
        last_day_of_previous_month = datetime.datetime(
            current_date.year, current_date.month, 1
        ) - datetime.timedelta(days=1)

        # Now we can get the month number of the previous month
        previous_month = last_day_of_previous_month.strftime("%m")

        VOFSTI = "ssb-vof-data-delt-stedfesting-prod/klargjorte-data/parquet"

        if geo_data == 'Yes':
            dataframes = []

            for year in range(2017, current_year_int + 1):
                file_path = f"{VOFSTI}/stedfesting-situasjonsuttak_p{year}-{previous_month}_v1.parquet"

                vof_df = dp.read_pandas(f"{file_path}")
                vof_gdf = gpd.GeoDataFrame(
                    vof_df,
                    geometry=gpd.points_from_xy(
                        vof_df["y_koordinat"],
                        vof_df["x_koordinat"],
                    ),
                    crs=25833,
                )

                vof_gdf = vof_gdf.rename(
                    columns={
                        "orgnrbed": "v_orgnr",
                        "org_nr": "orgnr_foretak",
                        "nace1_sn07": "naring",
                    }
                )

                vof_gdf = vof_gdf[
                    [
                        "v_orgnr",
                        "orgnr_foretak",
                        "naring",
                        "x_koordinat",
                        "y_koordinat",
                        "rute_100m",
                        "rute_1000m",
                        "geometry",
                    ]
                ]

                dataframes.append(vof_gdf)

            combined_gdf = pd.concat(dataframes, ignore_index=True)

            # Drop duplicate rows in the combined DataFrame
            combined_gdf = combined_gdf.drop_duplicates()

            # Merge with training_data
            training_data = pd.merge(training_data, combined_gdf, on="v_orgnr", how="left")
            
            return training_data
        
    if geo_data:
        print("collecting geodata")
        training_data = geo(training_data)

    
    temp = training_data.copy()
    
    temp.drop_duplicates(subset=['v_orgnr', 'id', 'year'], keep='first', inplace=True)
    
    merging_df = current_year_bad_oms[['v_orgnr', 'id', 'year', 'lopenr']]

    imputatable_df = pd.merge(merging_df, temp, on=['v_orgnr', 'id', 'year'], how='left')
      
    training_data = training_data[~training_data['v_orgnr'].isin(v_orgnr_list_for_imputering)]
    
    return current_year_good_oms, current_year_bad_oms, v_orgnr_list_for_imputering, training_data, imputatable_df, time_series_df, unique_id_list
    

