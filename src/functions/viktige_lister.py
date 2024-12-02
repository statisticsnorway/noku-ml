import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import gcsfs
import getpass
import os
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import sgis as sg
import dapla as dp
import datetime
from dapla.auth import AuthClient
from dapla import FileClient
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import requests
from pyjstat import pyjstat
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import interact, Dropdown, interactive_output, Button
from IPython.display import display, clear_output
import ipywidgets as widgets



import sys

sys.path.append("../functions")
import kommune_pop
import kommune_inntekt
import kpi
import ao
import kommune_translate
import kommune

fs = FileClient.get_gcs_file_system()
import numpy as np


import warnings

warnings.filterwarnings("ignore")

def collect_bedriftdata(aar):
    
    # Specify the columns you want to read
    columns_to_read = [
        "orgnr_foretak",
        "orgnr_bedrift",
        "naring",
        "kommune",
        "reg_type",
        "type",
        "naring_f",
        "navn",
        "giver_fnr",
        "giver_bnr",
        "omsetning",
        "sysselsetting_syss",
        "nopost_lonnskostnader",
        "nopost_driftskostnader",
        "produktinnsats",
        "produksjonsverdi",
        "ts_salgsint",
        "nopost_p4005",
        "nopost_p3000",
        "nopost_p3100",
        "nopost_p3200",
        "ts_forbruk",
        "bearbeidingsverdi",
        "bearbeidingsverdi_m"
    ]

    # Read only the specified columns
    bedrifter_pub = pd.read_parquet(
        f"gs://ssb-strukt-naering-data-produkt-prod/naringer/klargjorte-data/statistikkfiler/aar={aar}/statistikkfil_bedrifter_pub.parquet",
        filesystem=fs,
        columns=columns_to_read
    )


    # Filter rows where the first two characters of 'naring' are in ['45', '46', '47']
    bedrifter_pub = bedrifter_pub[
        bedrifter_pub['naring'].str[:2].isin(['45', '46', '47'])
    ]

    # Create 'fylke' column by extracting the first two characters of 'kommune'
    bedrifter_pub['fylke'] = bedrifter_pub['kommune'].str[:2]

    bedrifter_pub['antall'] = 1

    # Create a new variable 'no_salgint' as the sum of specified columns
    bedrifter_pub['no_salgint'] = (
        bedrifter_pub['nopost_p3000'] +
        bedrifter_pub['nopost_p3100'] +
        bedrifter_pub['nopost_p3200']
    )


    # Rename columns 'ts_salgsint' to 'salgsint' and 'ts_forbruk' to 'forbruk'
    bedrifter_pub = bedrifter_pub.rename(columns={
        "ts_salgsint": "salgsint",
        "ts_forbruk": "forbruk",
        "nopost_p4005": "p4005",
        "bearbeidingsverdi": "bearb",
        "bearbeidingsverdi_m": "bearb_m",
        "produksjonsverdi": "prodv",
        "produktinnsats": "prins",
        "nopost_driftskostnader": "drkost",
        "nopost_lonnskostnader": "lonnk",
    })


    bedrifter_pub['p4005_forb'] = bedrifter_pub['p4005'] - bedrifter_pub['forbruk']
    bedrifter_pub['dr_vk_lonn'] = bedrifter_pub['drkost'] - bedrifter_pub['p4005'] - bedrifter_pub['lonnk']
    bedrifter_pub['dr_forb_lonn'] = bedrifter_pub['drkost'] - bedrifter_pub['forbruk'] - bedrifter_pub['lonnk']
    bedrifter_pub['prins_vk_forb'] = bedrifter_pub['prins'] - bedrifter_pub['p4005_forb']
    bedrifter_pub['oms_salgsint'] = bedrifter_pub['omsetning'] - bedrifter_pub['salgsint'] 
    bedrifter_pub['nosalg_tssalg'] = bedrifter_pub['no_salgint'] - bedrifter_pub['salgsint'] 
    
    numerical_columns = [
        "antall",
        "omsetning",
        "no_salgint",
        "salgsint",
        "forbruk",
        "p4005",
        "bearb",
        "bearb_m",
        "prins",
        "prodv",
        "drkost",
        "lonnk",
        "p4005_forb",
        "dr_vk_lonn",
        "dr_forb_lonn",
        "prins_vk_forb",
        "oms_salgsint",
        "nosalg_tssalg",
    ]

    # Group by 'naring' and aggregate numerical columns
    bedrift_land = bedrifter_pub.groupby("naring")[numerical_columns].sum().reset_index()

    # Create a mask where at least one numerical column is negative
    mask_negative = bedrift_land[numerical_columns].lt(0).any(axis=1)

    # Filter bedrift_land to include only rows where the mask is True
    bedrift_land = bedrift_land.loc[mask_negative].reset_index(drop=True)

    bedrift_land = bedrift_land.sort_values(by='p4005_forb').reset_index(drop=True)
    
    # Group by 'naring' and 'fylke' and aggregate numerical columns
    bedrift_fylke = bedrifter_pub.groupby(["naring", "fylke"])[numerical_columns].sum().reset_index()

    # Create a mask where at least one numerical column is negative
    mask_negative = bedrift_fylke[numerical_columns].lt(0).any(axis=1)

    # Filter bedrift_land to include only rows where the mask is True
    bedrift_fylke = bedrift_fylke.loc[mask_negative].reset_index(drop=True)

    bedrift_fylke = bedrift_fylke.sort_values(by='p4005_forb').reset_index(drop=True)

    
    return bedrift_land, bedrift_fylke, bedrifter_pub


def nr_variabler_plot(bedrifter_pub):
    
    numerical_columns = [
        "antall",
        "omsetning",
        "no_salgint",
        "salgsint",
        "forbruk",
        "p4005",
        "bearb",
        "bearb_m",
        "prins",
        "prodv",
        "drkost",
        "lonnk",
        "p4005_forb",
        "dr_vk_lonn",
        "dr_forb_lonn",
        "prins_vk_forb",
        "oms_salgsint",
        "nosalg_tssalg",
    ]

    # Ensure numerical columns are numeric
    bedrifter_pub[numerical_columns] = bedrifter_pub[numerical_columns].apply(pd.to_numeric, errors='coerce')

    # Create widgets
    variable_dropdown = widgets.Dropdown(
        options=numerical_columns,
        value='p4005_forb',  # Set default variable
        description='Variabler:',
        disabled=False,
    )

    naring_options = sorted(bedrifter_pub['naring'].unique())
    
    naring_dropdown = widgets.SelectMultiple(
        options=naring_options,
        value=[],
        description='Næringer:',
        disabled=False,
        layout=widgets.Layout(height='200px')  # Set desired height here
    )
    # Display widgets
    # display(variable_dropdown, naring_dropdown)

    # Define plotting function with top 10 contributors
    def plot_negative_contributors(selected_variable, selected_naring_codes):
        # Filter data based on selected 'naring' codes
        if selected_naring_codes:
            filtered_df = bedrifter_pub[bedrifter_pub['naring'].isin(selected_naring_codes)].copy()
        else:
            filtered_df = bedrifter_pub.copy()

        # Filter rows where the selected variable is negative
        filtered_df = filtered_df[filtered_df[selected_variable] < 0]

        # Check if filtered DataFrame is empty
        if filtered_df.empty:
            print("No data available for the selected criteria.")
            return

        # Sort by the selected variable in ascending order
        filtered_df = filtered_df.sort_values(by=selected_variable)

        # Select top 10 contributors
        filtered_df = filtered_df.head(10)

        # Create the bar chart
        fig = px.bar(
            filtered_df,
            x=selected_variable,
            y='navn',
            orientation='h',
            hover_data=['orgnr_foretak', 'orgnr_bedrift', 'naring_f', 'navn'],
            labels={
                selected_variable: selected_variable,
                'navn': 'Bedrift Navn'
            },
            title=f'Topp 10 bedrifter som bidrar til negativ verdi {selected_variable}'
        )

        # Update layout for better readability
        fig.update_layout(
            yaxis=dict(autorange="reversed"),  # Largest negative at the top
            height=600
        )

        fig.show()

    # Create interactive plot
    interact(
        plot_negative_contributors,
        selected_variable=variable_dropdown,
        selected_naring_codes=naring_dropdown
    )




