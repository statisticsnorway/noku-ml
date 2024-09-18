from imports import *

def gather_data(year, start_year,  skjema_list):

    all_dfs = []  # List to store good dataframes for each year

    for current_year in range(start_year, year + 1):
        fjor = current_year - 1  # Previous year

        # List of skjema values
        # skjema_list = ['RA-0174-1',
        #                'RA-1403',
        #                'RA-1100',
        #                'RA-0255-1']

        # List to store DataFrames for each skjema in the current year
        year_dfs = []

        # Loop over each skjema and collect files
        for skjema in skjema_list:
            paths = [
                f for f in fs.glob(
                    f"gs://ssb-prod-noeku-data-produkt/eimerdb/nokubasen/skjemadata/aar={current_year}/skjema={skjema}/*"
                )
                if f.endswith(".parquet")
            ]

            if paths:
                # Load all parquet files for this skjema and year
                for path in paths:
                    skjema_df = pd.read_parquet(path, filesystem=fs)

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
                    ]

                    # Filter the DataFrame for the specified field values
                    skjema_df = skjema_df[skjema_df["feltnavn"].isin(felt_id_values)]

                    # Pivot the DataFrame
                    skjema_df = skjema_df.pivot_table(
                        index=["id", "radnr", "lopenr"],
                        columns="feltnavn",
                        values="feltverdi",
                        aggfunc="first",
                    )
                    skjema_df = skjema_df.reset_index()
                    skjema_df.columns = skjema_df.columns.str.lower()  # Convert column names to lower case

                    # Foretak level data is always when radnr = 0
                    foretak = skjema_df.loc[skjema_df["radnr"] <= 1]

                    foretak['year'] = current_year
                    foretak['skjema'] = skjema

                    # Append the DataFrame for this skjema to the list for the current year
                    year_dfs.append(foretak)

            else:
                print(f"No Parquet files found for skjema {skjema} in year {current_year}")

        # If there are any DataFrames for the current year, concatenate and append them
        if year_dfs:
            year_combined_df = pd.concat(year_dfs, ignore_index=True)
            all_dfs.append(year_combined_df)

    # Concatenate all DataFrames across all years
    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True)
    else:
        raise FileNotFoundError("No Parquet files found for any year.")
        
        
    foretak = df.loc[df["radnr"] == 0]
    bedrift = df.loc[df["radnr"] == 1]

    bedrift = bedrift.loc[bedrift["regtype"] == '01']



    foretak = foretak[['id', 'salgsint', 'nacef_5', 'forbruk', 'year']]

    bedrift = bedrift[['id', 'b_kommunenr', 'gjeldende_bdr_syss', 'gjeldende_driftsk_kr', 'gjeldende_lonn_kr', 'gjeldende_omsetn_kr', 'year']]

    training_data = pd.merge(bedrift, foretak, on=['id', 'year'], how='left')
    
    # create n3 as a substr of first 4 characters of nacef_5
    training_data['n3'] = training_data['nacef_5'].str[:4]
    training_data['n4'] = training_data['nacef_5'].str[:5]

    # change drype for omsetn_kr to float and for gjeldende_bdr_syss to float

    # replace ',' with '.' for omsetn_kr and gjeldende_bedr_syss
    training_data['gjeldende_omsetn_kr'] = training_data['gjeldende_omsetn_kr'].str.replace(',', '.')
    training_data['gjeldende_bdr_syss'] = training_data['gjeldende_bdr_syss'].str.replace(',', '.')
    training_data['salgsint'] = training_data['salgsint'].str.replace(',', '.')

    training_data['gjeldende_omsetn_kr'] = training_data['gjeldende_omsetn_kr'].astype(float)
    training_data['gjeldende_bdr_syss'] = training_data['gjeldende_bdr_syss'].astype(float)
    training_data['salgsint'] = training_data['salgsint'].astype(float)
    training_data['oms_per_syss'] = training_data['gjeldende_omsetn_kr'] / training_data['gjeldende_bdr_syss']
    training_data['salgsint_per_oms'] = training_data['salgsint'] / training_data['gjeldende_omsetn_kr']

    # fill nan and inf for oms_per_syss with 0
    training_data['oms_per_syss'] = training_data['oms_per_syss'].fillna(0)
    training_data['oms_per_syss'] = training_data['oms_per_syss'].replace([np.inf, -np.inf], 0)

    # convert Inf values to NaN
    training_data = training_data.replace([np.inf, -np.inf], np.nan)
    # count NaN values per column
    training_data.isna().sum()
    # training_data.shape

    # print rows that have at least one NaN value
    # test = training_data[training_data.isna().any(axis=1)]
    # test.head()

    # drop NaN values
    training_data = training_data.dropna()
    
    return training_data

def test_old_method(df):
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, r2_score

    # Reset index and make a copy for review
    df.reset_index(drop=True, inplace=True)


    def find_nearest_neighbor(df, index):
        current_id = df.at[index, 'id']  # Get the 'id' of the current row
        row = df.iloc[index]
        # Filter by nacef_5, then n4, then n3 if no matches found
        filters = ['nacef_5', 'n4', 'n3']
        for f in filters:
            # Exclude the current row based on 'id'
            subset = df[(df[f] == row[f]) & (df['id'] != current_id)]
            if not subset.empty:
                # Find the closest neighbor based on oms_per_syss
                closest_idx = (subset['oms_per_syss'] - row['oms_per_syss']).abs().idxmin()
                return subset.loc[closest_idx]
        return None  # No neighbor found

    def predict_salgsint(df):
        predictions = []
        givers = []  # Track the 'id' of the neighbor
        for i in range(len(df)):
            neighbor = find_nearest_neighbor(df, i)
            if neighbor is not None:
                predicted_salgsint = neighbor['salgsint_per_oms'] * df.at[i, 'gjeldende_omsetn_kr']
                predictions.append(predicted_salgsint)
                givers.append(neighbor['id'])  # Store the 'id' of the neighbor
            else:
                predictions.append(None)  # Handle case with no neighbors found
                givers.append(None)
        return predictions, givers

        # Add predictions and givers to the dataframe
        
    df['predicted_salgsint'], df['giver'] = predict_salgsint(df)

    # Drop rows with no predictions
    df.dropna(subset=['predicted_salgsint', 'giver'], inplace=True)

    # Replace inf and NaN values to ensure clean data for analysis
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['predicted_salgsint', 'salgsint'], inplace=True)

    # Handle division by zero for percentage error
    df = df[df['salgsint'] != 0]

    # Calculate MAE and R-squared
    mae = mean_absolute_error(df['salgsint'], df['predicted_salgsint'])
    r_squared = r2_score(df['salgsint'], df['predicted_salgsint'])

    # Calculate percentage error
    # df['percentage_error'] = 100 * (df['salgsint'] - df['predicted_salgsint']) / df['salgsint']
    df['percentage_error'] = 100 * np.abs(df['salgsint'] - df['predicted_salgsint']) / df['salgsint']
    df['percentage_error'].replace([np.inf, -np.inf], 100, inplace=True)
    
    percentage_error_avg = np.mean(np.abs(df['percentage_error']))

    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared: {r_squared}")
    print(f"Average Percentage Error: {percentage_error_avg}%")

    # Plotting Predicted vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(df['salgsint'], df['predicted_salgsint'], alpha=0.5)
    plt.title('Predicted vs Actual Sales')
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.plot([df['salgsint'].min(), df['salgsint'].max()], 
             [df['salgsint'].min(), df['salgsint'].max()], 
             'k--', lw=4)
    plt.show()

    # Residual plot (errors between actual and predicted values)
    df['residuals'] = df['salgsint'] - df['predicted_salgsint']
    plt.figure(figsize=(10, 6))
    plt.scatter(df['predicted_salgsint'], df['residuals'], alpha=0.5)
    plt.axhline(y=0, color='k', linestyle='--', lw=2)
    plt.title('Residual Plot')
    plt.xlabel('Predicted Sales')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.show()

    return df


def bootstrap(df, number_iterations):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    def find_nearest_neighbor(X, row):
        filters = ['nacef_5', 'n4', 'n3']
        for f in filters:
            subset = X[(X[f] == row[f]) & (X['id'] != row['id'])]
            if not subset.empty:
                closest_idx = (subset['oms_per_syss'] - row['oms_per_syss']).abs().idxmin()
                return subset.loc[closest_idx]
        return None

    def predict_salgsint(X, Y):
        predictions = []
        givers = []
        for _, row in Y.iterrows():
            neighbor = find_nearest_neighbor(X, row)
            if neighbor is not None:
                predicted_salgsint = neighbor['salgsint_per_oms'] * row['gjeldende_omsetn_kr']
                predictions.append(predicted_salgsint)
                givers.append(neighbor['id'])
            else:
                predictions.append(None)
                givers.append(None)
        return predictions, givers

    def monte_carlo_simulation(data, iterations=number_iterations):
        detailed_results = []
        summary_results = []
        for iteration in range(iterations):
            Y = data.sample(frac=0.8, random_state=iteration)
            X = data.drop(Y.index)

            Y['predicted_salgsint'], Y['giver'] = predict_salgsint(X, Y)

            # Clean inf and NaN values for calculations
            Y.replace([np.inf, -np.inf], np.nan, inplace=True)
            Y.dropna(subset=['predicted_salgsint', 'salgsint'], inplace=True)


            # Calculate percentage difference, and replace 'inf' values with 100%
            Y['percentage_difference'] = 100 * np.abs(Y['salgsint'] - Y['predicted_salgsint']) / Y['salgsint']
            Y['percentage_difference'].replace([np.inf, -np.inf], 100, inplace=True)
            Y['iteration'] = iteration  # Tag each row with the iteration number

            detailed_results.append(Y)

            # Evaluate the results
            valid_rows = Y.dropna(subset=['percentage_difference'])
            mae = mean_absolute_error(valid_rows['salgsint'], valid_rows['predicted_salgsint'])
            r_squared = r2_score(valid_rows['salgsint'], valid_rows['predicted_salgsint'])
            avg_percentage_diff = valid_rows['percentage_difference'].mean()

            summary_results.append({'MAE': mae, 'R_squared': r_squared, 'Avg_Percentage_Diff': avg_percentage_diff})

        # Concatenate all results into a single DataFrame and sort
        detailed_df = pd.concat(detailed_results)
        detailed_df = detailed_df.sort_values(by='percentage_difference', ascending=False)

        return summary_results, detailed_df

    # Running the Monte Carlo Simulation
    summary_stats, results_df = monte_carlo_simulation(df)

    # Display summary statistics
    avg_mae = np.mean([result['MAE'] for result in summary_stats])
    avg_r_squared = np.mean([result['R_squared'] for result in summary_stats])
    avg_percentage_diff = np.mean([result['Avg_Percentage_Diff'] for result in summary_stats])

    print(f"Average Mean Absolute Error: {avg_mae}")
    print(f"Average R-squared: {avg_r_squared}")
    print(f"Average Percentage Difference: {avg_percentage_diff}%")

    # Optionally, plot the distribution of MAE across simulations
    mae_values = [result['MAE'] for result in summary_stats]
    plt.hist(mae_values, bins=10, alpha=0.75)
    plt.title('Distribution of MAE across Simulations')
    plt.xlabel('Mean Absolute Error')
    plt.ylabel('Frequency')
    plt.show()

    # Adding Residual Plot
    results_df['residuals'] = results_df['salgsint'] - results_df['predicted_salgsint']

    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['predicted_salgsint'], results_df['residuals'], alpha=0.5)
    plt.axhline(y=0, color='k', linestyle='--', lw=2)
    plt.title('Residual Plot')
    plt.xlabel('Predicted Sales')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.show()
    

def evaluate_varehandel(current_year, start_year):
    
    import pandas as pd
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score
    import matplotlib.pyplot as plt
    
    all_dfs = []
    all_stat_dfs = []

    for current_year in range(start_year, current_year + 1):

        fil_path = [
            f
            for f in fs.glob(
                f"gs://ssb-prod-noeku-data-produkt/statistikkfiler/g{current_year}/statistikkfil_bedrifter_pub.parquet"
            )
            if f.endswith(".parquet")
        ]

        # Use the ParquetDataset to read multiple files
        dataset = pq.ParquetDataset(fil_path, filesystem=fs)
        table = dataset.read()

        # Convert to Pandas DataFrame
        skjema = table.to_pandas()

        skjema = skjema.reset_index()
        skjema.columns = skjema.columns.str.lower()  # Convert column names to lower case

        skjema['year'] = current_year

        # Apply the filters
        skjema = skjema[
            (skjema['reg_type_f'] == '02') & 
            (skjema['naring'].str[:2].isin(['45', '46', '47']))
        ]

        skjema = skjema[['omsetning', 'naring', 'driftskost_kr', 'reg_type_f', 'ts_salgsint', 'enhets_id', 'sysselsetting_syss', 'year', 'gjeldende_lonn_kr', 'kommune']]

        # rename enhets_id to 'id'
        skjema = skjema.rename(columns={'enhets_id': 'id',
                                        'sysselsetting_syss': 'gjeldende_bdr_syss',
                                        'omsetning': 'gjeldende_omsetn_kr',
                                        'naring': 'nacef_5',
                                        'ts_salgsint': 'salgsint',
                                        'driftskost_kr': 'gjeldende_driftsk_kr',
                                        'kommune': 'b_kommunenr',
                                        })

        skjema['n3'] = skjema['nacef_5'].str[:4]
        skjema['n4'] = skjema['nacef_5'].str[:5]

        # replace ',' with '.' for omsetn_kr and gjeldende_bedr_syss
        # skjema['gjeldende_omsetn_kr'] = skjema['gjeldende_omsetn_kr'].str.replace(',', '.')
        # skjema['gjeldende_bdr_syss'] = skjema['gjeldende_bdr_syss'].str.replace(',', '.')
        # skjema['salgsint'] = skjema['salgsint'].str.replace(',', '.')

        skjema['gjeldende_omsetn_kr'] = skjema['gjeldende_omsetn_kr'].astype(float)
        skjema['gjeldende_bdr_syss'] = skjema['gjeldende_bdr_syss'].astype(float)
        skjema['salgsint'] = skjema['salgsint'].astype(float)
        skjema['oms_per_syss'] = skjema['gjeldende_omsetn_kr'] / skjema['gjeldende_bdr_syss']
        skjema['salgsint_per_oms'] = skjema['salgsint'] / skjema['gjeldende_omsetn_kr']

        # fill nan and inf for oms_per_syss with 0
        skjema['oms_per_syss'] = skjema['oms_per_syss'].fillna(0)
        skjema['oms_per_syss'] = skjema['oms_per_syss'].replace([np.inf, -np.inf], 0)

        skjema = skjema[skjema['gjeldende_bdr_syss'] != 0]

        # Add the filtered DataFrame to the list
        all_stat_dfs.append(skjema)

    # Concatenate all DataFrames into one
    stat = pd.concat(all_stat_dfs, ignore_index=True)
    
    for current_year in range(start_year, current_year + 1):
        fjor = current_year - 1  # Previous year

        # skjema_list = ['RA-0174-1', 'RA-0174A3', 'RA-0827A3']
        # skjema_list = 'RA-0174-1'

        skjema_list = 'RA-0174-1'

        # Convert skjema_list to a tuple to use in the SQL query

        fil_path = [
            f
            for f in fs.glob(
                f"gs://ssb-prod-noeku-data-produkt/eimerdb/nokubasen/skjemadata/aar={current_year}/skjema={skjema_list}/*"
            )
            if f.endswith(".parquet")
        ]

        # Assuming there's only one file in fil_path
        if fil_path:
            skjema = pd.read_parquet(fil_path[0], filesystem=fs)
        else:
            raise FileNotFoundError(f"No Parquet files found for year {current_year}")

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
        ]

        # Filter the DataFrame for the specified field values
        skjema = skjema[skjema["feltnavn"].isin(felt_id_values)]

        # Pivot the DataFrame
        skjema = skjema.pivot_table(
            index=["id", "radnr", "lopenr"],
            columns="feltnavn",
            values="feltverdi",
            aggfunc="first",
        )
        skjema = skjema.reset_index()
        skjema.columns = skjema.columns.str.lower()  # Convert column names to lower case

        # Foretak level data is always when radnr = 0
        foretak = skjema.loc[skjema["radnr"] <= 1]

        foretak['year'] = current_year


        # Create the DataFrames

        all_dfs.append(foretak)

    df = pd.concat(all_dfs, ignore_index=True)

    foretak = df.loc[df["radnr"] == 0]
    bedrift = df.loc[df["radnr"] == 1]

    bedrift = bedrift.loc[bedrift["regtype"] == '01']



    foretak = foretak[['id', 'salgsint', 'nacef_5', 'forbruk', 'year']]

    bedrift = bedrift[['id', 'b_kommunenr', 'gjeldende_bdr_syss', 'gjeldende_driftsk_kr', 'gjeldende_lonn_kr', 'gjeldende_omsetn_kr', 'year']]

    training_data = pd.merge(bedrift, foretak, on=['id', 'year'], how='left')

    # create n3 as a substr of first 4 characters of nacef_5
    training_data['n3'] = training_data['nacef_5'].str[:4]
    training_data['n4'] = training_data['nacef_5'].str[:5]
    training_data['n2'] = training_data['nacef_5'].str[:2]

    # change drype for omsetn_kr to float and for gjeldende_bdr_syss to float

    # replace ',' with '.' for omsetn_kr and gjeldende_bedr_syss
    training_data['gjeldende_omsetn_kr'] = training_data['gjeldende_omsetn_kr'].str.replace(',', '.')
    training_data['gjeldende_bdr_syss'] = training_data['gjeldende_bdr_syss'].str.replace(',', '.')
    training_data['salgsint'] = training_data['salgsint'].str.replace(',', '.')

    training_data['gjeldende_omsetn_kr'] = training_data['gjeldende_omsetn_kr'].astype(float)
    training_data['gjeldende_bdr_syss'] = training_data['gjeldende_bdr_syss'].astype(float)
    training_data['salgsint'] = training_data['salgsint'].astype(float)
    training_data['oms_per_syss'] = training_data['gjeldende_omsetn_kr'] / training_data['gjeldende_bdr_syss']
    training_data['salgsint_per_oms'] = training_data['salgsint'] / training_data['gjeldende_omsetn_kr']

    # fill nan and inf for oms_per_syss with 0
    training_data['oms_per_syss'] = training_data['oms_per_syss'].fillna(0)
    training_data['oms_per_syss'] = training_data['oms_per_syss'].replace([np.inf, -np.inf], 0)

    # convert Inf values to NaN
    training_data = training_data.replace([np.inf, -np.inf], np.nan)

    # if gjeldende_bdr_syss is 0 then delete
    training_data = training_data[training_data['gjeldende_bdr_syss'] != 0]

    # print rows that have at least one NaN value
    # test = training_data[training_data.isna().any(axis=1)]
    # test.head()

    # drop NaN values
    training_data = training_data.dropna()
    training_data = training_data[training_data['nacef_5'].str[:2].isin(['45', '46', '47'])]

    def find_nearest_neighbors(X, row, num_neighbors=3):
        neighbors_list = pd.DataFrame()  # Initialize an empty DataFrame

        # Check each level starting from most specific to least specific
        for category in ['nacef_5', 'n4', 'n3']:
            subset = X[(X[category] == row[category]) & (X['id'] != row['id'])]
            if not subset.empty:
                # Normalize 'oms_per_syss' within the subset
                scaler = StandardScaler()
                distances = scaler.fit_transform(subset[['oms_per_syss']])
                target = scaler.transform([[row['oms_per_syss']]])

                # Use NearestNeighbors to find the closest entries
                nn = NearestNeighbors(n_neighbors=min(num_neighbors, len(subset)))
                nn.fit(distances)
                distances, indices = nn.kneighbors(target)

                # Select the nearest entries
                if indices[0].size > 0:
                    neighbors = subset.iloc[indices[0]]
                    if neighbors_list.empty:
                        neighbors_list = neighbors
                    else:
                        neighbors_list = pd.concat([neighbors_list, neighbors])

                if len(neighbors_list) >= num_neighbors:
                    break

        return neighbors_list.head(num_neighbors)  # Ensure only the top n neighbors are returned

    def predict_salgsint(X, Y):
        predictions = []
        for _, row in Y.iterrows():
            neighbors = find_nearest_neighbors(X, row)
            if not neighbors.empty:
                avg_salgsint_per_oms = neighbors['salgsint_per_oms'].mean()
                predicted_salgsint = avg_salgsint_per_oms * row['gjeldende_omsetn_kr']
                predictions.append(predicted_salgsint)
            else:
                predictions.append(None)
        return predictions

    # Assuming 'stat' and 'training_data' are already loaded and cleaned
    X = stat.copy()
    Y = training_data.copy()

    # Predict 'salgsint' for all rows in 'training_data' using 'stat'
    Y['predicted_salgsint'] = predict_salgsint(X, Y)

    # Calculate percentage difference
    # Y['percentage_difference'] = 100 * np.abs(Y['salgsint'] - Y['predicted_salgsint']) / Y['salgsint']
    Y['percentage_difference'] = 100 * np.abs(Y['salgsint'] - Y['predicted_salgsint']) / Y['salgsint']
    Y['percentage_difference'].replace([np.inf, -np.inf], 100, inplace=True)
    valid_rows = Y.dropna(subset=['percentage_difference'])

    # Evaluate the results
    mae = mean_absolute_error(valid_rows['salgsint'], valid_rows['predicted_salgsint'])
    r_squared = r2_score(valid_rows['salgsint'], valid_rows['predicted_salgsint'])

    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared: {r_squared}")

    # Plotting Predicted vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(valid_rows['salgsint'], valid_rows['predicted_salgsint'], alpha=0.5)
    plt.title('Predicted vs Actual Sales')
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.plot([valid_rows['salgsint'].min(), valid_rows['salgsint'].max()], 
             [valid_rows['salgsint'].min(), valid_rows['salgsint'].max()], 
             'k--', lw=2)
    plt.show()
    
    # Adding Residual Plot
    valid_rows['residuals'] = valid_rows['salgsint'] - valid_rows['predicted_salgsint']

    plt.figure(figsize=(10, 6))
    plt.scatter(valid_rows['predicted_salgsint'], valid_rows['residuals'], alpha=0.5)
    plt.axhline(y=0, color='k', linestyle='--', lw=2)
    plt.title('Residual Plot (Actual - Predicted)')
    plt.xlabel('Predicted Sales Intensity')
    plt.ylabel('Residuals')
    plt.show()
    
    eval_dfs = []

    for current_year in range(start_year, current_year + 1):

        fil_path = [
            f
            for f in fs.glob(
                f"gs://ssb-prod-noeku-data-produkt/statistikkfiler/g{current_year}/statistikkfil_bedrifter_pub.parquet"
            )
            if f.endswith(".parquet")
        ]

        # Use the ParquetDataset to read multiple files
        dataset = pq.ParquetDataset(fil_path, filesystem=fs)
        table = dataset.read()

        # Convert to Pandas DataFrame
        skjema = table.to_pandas()

        skjema = skjema.reset_index()
        skjema.columns = skjema.columns.str.lower()  # Convert column names to lower case

        skjema['year'] = current_year

        # Apply the filters
        skjema = skjema[
            (skjema['reg_type_f'] == '01') & 
            (skjema['naring'].str[:2].isin(['45', '46', '47']))
        ]

        skjema = skjema[['omsetning', 'naring', 'driftskost_kr', 'reg_type_f', 'ts_salgsint', 'enhets_id', 'sysselsetting_syss', 'year', 'gjeldende_lonn_kr', 'kommune']]

        # rename enhets_id to 'id'
        skjema = skjema.rename(columns={'enhets_id': 'id',
                                        'sysselsetting_syss': 'gjeldende_bdr_syss',
                                        'omsetning': 'gjeldende_omsetn_kr',
                                        'naring': 'nacef_5',
                                        'ts_salgsint': 'salgsint',
                                        'driftskost_kr': 'gjeldende_driftsk_kr',
                                        'kommune': 'b_kommunenr',
                                        })

        skjema['n3'] = skjema['nacef_5'].str[:4]
        skjema['n4'] = skjema['nacef_5'].str[:5]

        # replace ',' with '.' for omsetn_kr and gjeldende_bedr_syss
        # skjema['gjeldende_omsetn_kr'] = skjema['gjeldende_omsetn_kr'].str.replace(',', '.')
        # skjema['gjeldende_bdr_syss'] = skjema['gjeldende_bdr_syss'].str.replace(',', '.')
        # skjema['salgsint'] = skjema['salgsint'].str.replace(',', '.')

        skjema['gjeldende_omsetn_kr'] = skjema['gjeldende_omsetn_kr'].astype(float)
        skjema['gjeldende_bdr_syss'] = skjema['gjeldende_bdr_syss'].astype(float)
        skjema['salgsint'] = skjema['salgsint'].astype(float)
        skjema['oms_per_syss'] = skjema['gjeldende_omsetn_kr'] / skjema['gjeldende_bdr_syss']
        skjema['salgsint_per_oms'] = skjema['salgsint'] / skjema['gjeldende_omsetn_kr']

        # fill nan and inf for oms_per_syss with 0
        skjema['oms_per_syss'] = skjema['oms_per_syss'].fillna(0)
        skjema['oms_per_syss'] = skjema['oms_per_syss'].replace([np.inf, -np.inf], 0)

        skjema = skjema[skjema['gjeldende_bdr_syss'] != 0]

        # Add the filtered DataFrame to the list
        eval_dfs.append(skjema)

    # Concatenate all DataFrames into one
    eval_df = pd.concat(eval_dfs, ignore_index=True)

    eval_df = eval_df[['id', 'salgsint', 'year', 'n3']]
    
    Y['predicted_salgsint'].fillna(0, inplace=True)
    
    merge1 = Y[['id', 'year', 'predicted_salgsint']]
    
    prelim_df = pd.merge(merge1, eval_df, on=['id', 'year'], how='left')
    
    prelim_df_agg = prelim_df.groupby(['n3', 'year']).agg({'predicted_salgsint': 'sum', 'salgsint': 'sum'}).reset_index()
    
    prelim_df_agg['diff'] = prelim_df_agg['predicted_salgsint'] - prelim_df_agg['salgsint']
    prelim_df_agg['diff_percent'] = 100 * prelim_df_agg['diff'] / prelim_df_agg['salgsint']
    prelim_df_agg['diff_percent'] = prelim_df_agg['diff_percent'].apply(lambda x: f"{x:.2f}%")
    
    return prelim_df_agg
   
def cumulative_histogram():
    import pandas as pd
    from ipywidgets import interact, widgets
    import plotly.graph_objects as go

    current_year = 2022

    fil_path = [
        f
        for f in fs.glob(
            f"gs://ssb-prod-noeku-data-produkt/statistikkfiler/g{current_year}/statistiske_foretak_foretak.parquet"
        )
        if f.endswith(".parquet")
    ]

    # Use the ParquetDataset to read multiple files
    dataset = pq.ParquetDataset(fil_path, filesystem=fs)
    table = dataset.read()

    # Convert to Pandas DataFrame
    skjema = table.to_pandas()

    skjema = skjema[['orgnr_foretak', 'Omsetning', 'naring']]

    skjema['n3'] = skjema['naring'].str.slice(0, 4)
    
    df = skjema.copy()

    # Identify numerical columns (excluding 'orgnr_foretak' if necessary)
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    numerical_columns = [col for col in numerical_columns if col != 'orgnr_foretak']

    if not numerical_columns:
        print("No numerical columns available in the DataFrame for plotting.")
        return

    def update_plot(variable, naring, source_data):
        # Filter and prepare data
        data = source_data[source_data['n3'] == naring]

        # Check if data is available after filtering
        if data.empty:
            print(f"No data available for Næring '{naring}'.")
            return

        # Sort data by the selected variable
        sorted_data = data.sort_values(by=variable, ascending=False).reset_index(drop=True)

        # Calculate cumulative sum and percentage
        sorted_data['cumulative'] = sorted_data[variable].cumsum()
        total = sorted_data[variable].sum()
        if total == 0:
            sorted_data['cumulative_pct'] = 0
        else:
            sorted_data['cumulative_pct'] = 100 * sorted_data['cumulative'] / total

        sorted_data['rank'] = range(1, len(sorted_data) + 1)

        # Get max value for dynamic axis range
        max_value = sorted_data[variable].max()

        # Plot configuration
        fig = go.Figure()

        # Add bar chart for variable values
        fig.add_trace(go.Bar(
            x=sorted_data['rank'],
            y=sorted_data[variable],
            name=f'{variable} per Business',
            marker=dict(color='blue'),
            yaxis='y2'
        ))

        # Add line chart for cumulative percentage
        fig.add_trace(go.Scatter(
            x=sorted_data['rank'],
            y=sorted_data['cumulative_pct'],
            name='Cumulative %',
            marker=dict(color='red'),
            mode='lines+markers',
            yaxis='y1'  # Explicitly associate with primary y-axis
        ))

        # Layout with dual y-axes and dynamic scaling
        fig.update_layout(
            title=f"Cumulative Distribution of {variable} for {naring}",
            xaxis_title="Business Rank",
            yaxis=dict(
                title="Cumulative Percentage (%)",
                range=[0, 100]  # Keep percentage axis fixed from 0 to 100
            ),
            yaxis2=dict(
                title=f"{variable} Value",
                overlaying='y',
                side='right',
                showgrid=False,
                range=[0, max_value * 1.1]  # Dynamic range based on data
            ),
            hovermode='x',
            width=900,
            height=600
        )

        fig.show()

    # Widgets for interactive selection
    variable_selector = widgets.Dropdown(
        options=numerical_columns,
        value=numerical_columns[0],
        description='Variable:'
    )

    sorted_n3_options = sorted(df['n3'].dropna().unique())

    naring_selector = widgets.Dropdown(
        options=sorted_n3_options,
        description='Næring:'
    )

    # Integration with widgets
    interact(update_plot,
             variable=variable_selector,
             naring=naring_selector,
             source_data=widgets.fixed(df))





