# def kommune():
#     import pandas as pd
#     import sgis as sg

#     testdatasti = "ssb-prod-dapla-felles-data-delt/GIS/testdata"
#     kommuner = sg.read_geopandas(f"{testdatasti}/enkle_kommuner.parquet")
#     pop = pd.read_excel("Befolkning.xlsx")
#     inntekt = pd.read_excel("inntekt.xlsx")
#     pop["KOMMUNENR"] = pop["KOMMUNENR"].astype("object")
#     inntekt["KOMMUNENR"] = inntekt["KOMMUNENR"].astype("object")

#     kommuner["KOMMUNENR"] = kommuner["KOMMUNENR"].str.replace('"', "").astype(str)
#     pop["KOMMUNENR"] = pop["KOMMUNENR"].astype(str)
#     pop["KOMMUNENR"] = pop["KOMMUNENR"].astype(str).str.zfill(4)

#     inntekt["KOMMUNENR"] = inntekt["KOMMUNENR"].astype(str)
#     inntekt["KOMMUNENR"] = inntekt["KOMMUNENR"].astype(str).str.zfill(4)

#     kommuner = kommuner.merge(pop, on="KOMMUNENR", how="left")
#     kommuner = kommuner.merge(inntekt, on="KOMMUNENR", how="left")

#     return kommuner


def kommune(variable, naring, year, df):
    
    import pandas as pd
    import sgis as sg

    testdatasti = "ssb-prod-dapla-felles-data-delt/GIS/testdata"
    kommuner = sg.read_geopandas(f"{testdatasti}/enkle_kommuner.parquet")
    
    df = df[df['year'] == year]
    
    # Group by 'n3' and 'b_kommunenr', and then sum up the 'oms' values
    aggregated_df = df.groupby(['n3', 'kommunenr'])[variable].sum().reset_index()

    # Filter the aggregated DataFrame where 'n3' equals '47.2'
    filtered_df = aggregated_df[aggregated_df['n3'] == naring]

    # Display the first few rows of the filtered DataFrame
    filtered_df = filtered_df.rename(columns={
        'kommunenr': 'KOMMUNENR'
    })

    filtered_df = filtered_df[['KOMMUNENR', variable]]

    kommuner["KOMMUNENR"] = kommuner["KOMMUNENR"].str.replace('"', "").astype(str)
    filtered_df["KOMMUNENR"] = filtered_df["KOMMUNENR"].astype(str)
    filtered_df["KOMMUNENR"] = filtered_df["KOMMUNENR"].astype(str).str.zfill(4)

    kommuner = pd.merge(kommuner, filtered_df, on="KOMMUNENR", how="left")

    # fill NaN oms with 0
    kommuner[variable] = kommuner[variable].fillna(0)


    return kommuner

def get_coordinates(df):
    
    import pandas as pd
    import geopandas as gpd
    import sgis as sg
    import dapla as dp
    import datetime
    from dapla.auth import AuthClient
    from dapla import FileClient
    
    # Add geographical data:
    
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

    df = pd.merge(df, combined_gdf, on="v_orgnr", how="left")
    
    merged_gdf = gpd.GeoDataFrame(df, geometry="geometry")
    merged_gdf = merged_gdf.dropna(subset=["x_koordinat", "y_koordinat"])
    
    return merged_gdf