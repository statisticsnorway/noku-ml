from imports import *
import plotly.graph_objects as go

def holymoly(year, start_year, words):
    
    happy_days = []  # List to store dataframes for each year
    
    # Correctly define the skjema_list as a list of strings
    skjema_list = [
        'RA-1403',
        'RA-0255-1',
        'RA-0174-1',
        'RA-0174-4',
        'RA-0351-2',
        'RA-0351-3',
        'RA-0351-4',
        'RA-0351-11',
        'RA-0351-5',
        'RA-0351-6',
        'RA-0481',
        'RA-0351-1',
        'RA-0351-10',
        'RA-0351-13',
        'RA-0351-14',
        'RA-1100',
        'RA-1101',
        'RA-0351-21',
        'RA-1407',
        'RA-0809'
    ]

    felt_id_values = [
        "KOMMENTARER",
        "KOMMENTAR",
        "TMP_AKTIVITET", 
        "KOMMENTAR01R",
        "KOMMENTAR01N",
        "KOMMENTAR01H",
        "KOMMENTAR02O",
        "KOMMENTAR02S",
        "KOMMENTAR03K",
        "KOMMENTAR04I",
        "KOMMENTAR04IKT",
        "KOMMENTAR05V",
        "KOMMENTAR06I",
        "KOMMENTAR07I",
        "KOMMENTAR08I",
        "KOMMENTARER_INT",
        "KOMMENTAR_EKST",
        "KOMMENTAR_INT"
        "TMP_KONT_NAVN"
        "TMP_ANNEN_NAERINGSTEKST"
        "VASK_KOMMENTAR"
        "ORGNR_N_1",
        "NACEF_5",
        "ID",
        "ARBEIDSFELLESSKAP_NAVN",
    ]

    for current_year in range(start_year, year + 1):
        fjor = current_year - 1  # Previous year

        for skjema in skjema_list:  # Loop through each skjema in the list
            fil_path = [
                f
                for f in fs.glob(
                    f"gs://ssb-prod-noeku-data-produkt/eimerdb/nokubasen/skjemadata/aar={current_year}/skjema={skjema}/*"
                )
                if f.endswith(".parquet")
            ]

            if not fil_path:
                continue  # Skip if no files are found



            # Use the ParquetDataset to read multiple files with filters
            filters = [
                ('radnr', '=', 0),
                ('feltnavn', 'in', felt_id_values)
            ]

            dataset = pq.ParquetDataset(fil_path, filesystem=fs, filters=filters)
            table = dataset.read()

            # Convert to Pandas DataFrame
            skjema_df = table.to_pandas()

            # Pivot the DataFrame
            skjema_df = skjema_df.pivot_table(
                index=["id", "radnr", "lopenr"],
                columns="feltnavn",
                values="feltverdi",
                aggfunc="first",
            )
            skjema_df = skjema_df.reset_index()

            skjema_df['aar'] = current_year

            skjema_df.columns = skjema_df.columns.str.lower()  # Convert column names to lower case

            happy_days.append(skjema_df)

    # Concatenate all DataFrames into a single DataFrame
    happy_days = pd.concat(happy_days, ignore_index=True)
    
    def count_words_in_row(row, words):
        count = 0
        exclamation_count = 0
        for word in words:
            for col in [
                "kommentar01h",
                "kommentar01n",
                "kommentar02o",
                "kommentar03k",
                "kommentar04i",
                "kommentar04ikt",
                "kommentar05v",
                "kommentarer",
                "kommentarer_int",
                "tmp_aktivitet",
                "kommentar01r",
                "arbeidsfellesskap_navn",
                "kommentar_ekst",
                "kommentar02s",
                "kommentarer_int",
                "kommentar01r",
            ]:
                # Ensure the cell is a string; if not, treat it as an empty string
                cell_value = str(row[col]) if pd.notna(row[col]) else ""
                count += cell_value.lower().split().count(word)
                exclamation_count += cell_value.count("!")
        return count, exclamation_count


    # Apply the function to each row to create the 'count' and 'exclamation_count' columns
    happy_days[["count", "exclamation_count"]] = happy_days.apply(
        lambda row: pd.Series(count_words_in_row(row, words)), axis=1
    )
    
    happy_days['n3'] = happy_days['nacef_5'].str[:4]
    happy_days['n2'] = happy_days['nacef_5'].str[:2]
    
    happy_days = happy_days.reset_index(drop=True)

    # sort exclamation_count and count
    happy_days = happy_days.sort_values(by=["exclamation_count", "count"], ascending=False)
    
    # group by n3 and sum count and exclamation_count
    happy_days_overall = happy_days.groupby('n3').agg({'count': 'sum', 'exclamation_count': 'sum'}).reset_index()
    happy_days_overall_n2 = happy_days.groupby('n2').agg({'count': 'sum', 'exclamation_count': 'sum'}).reset_index()

    # group by n3 and aar and sum count and exclamation_count
    happy_days_by_year = happy_days.groupby(['aar', 'n3']).agg({'count': 'sum', 'exclamation_count': 'sum'}).reset_index()
    happy_days_by_year_n2 = happy_days.groupby(['aar', 'n2']).agg({'count': 'sum', 'exclamation_count': 'sum'}).reset_index()
    
    happy_days_overall = happy_days_overall.sort_values(by='count', ascending=False)
    happy_days_by_year = happy_days_by_year.sort_values(by='count', ascending=False)
    happy_days_overall_n2 = happy_days_overall_n2.sort_values(by='count', ascending=False)
    happy_days_by_year_n2 = happy_days_by_year_n2.sort_values(by='count', ascending=False)
    
    everything = happy_days.groupby(['aar']).agg({'count': 'sum', 'exclamation_count': 'sum'}).reset_index()
    
#     def static_barchart(df):
#         # Filter the DataFrame for the year 2022

#         # if count = 0 then delete
#         df = df[df['count'] > 0]

#         # Sort the DataFrame by 'count' in descending order
#         df = df.sort_values(by='count', ascending=False)

#         # Keep only the top 10 results
#         df = df.head(10)

#         # Create a horizontal bar chart with a predefined color sequence
#         fig = px.bar(
#             df,
#             x='count',      # Set 'count' as the x-axis
#             y='n2',         # Set 'n2' as the y-axis
#             color='n2',     # Color bars by 'n2'
#             color_discrete_sequence=px.colors.sequential.Viridis,  # Use the Viridis color sequence
#             orientation='h',  # Make the bars horizontal
#             height=1200,    # Set height of the plot
#             width=900,      # Set width of the plot
#             title='The total amount of swear words used for n2 from 2017 to 2021'  # Add title
#         )

#         # Ensure y-axis categories are sorted by 'count'
#         fig.update_yaxes(categoryorder='total ascending')

#         # Show the plot
#         fig.show()

#     def static_barchart_n3(df):
#         # Filter the DataFrame for the year 2022

#         # if count = 0 then delete
#         df = df[df['count'] > 0]

#         # Sort the DataFrame by 'count' in descending order
#         df = df.sort_values(by='count', ascending=False)

#         # Keep only the top 10 results
#         df = df.head(10)

#         # Create a horizontal bar chart with a predefined color sequence
#         fig = px.bar(
#             df,
#             x='count',      # Set 'count' as the x-axis
#             y='n3',         # Set 'n2' as the y-axis
#             color='n3',     # Color bars by 'n2'
#             color_discrete_sequence=px.colors.sequential.Viridis,  # Use the Viridis color sequence
#             orientation='h',  # Make the bars horizontal
#             height=1200,    # Set height of the plot
#             width=900,      # Set width of the plot
#             title='The total amount of swear words used per n3 from 2017 o 2021'  # Add title
#         )

#         # Ensure y-axis categories are sorted by 'count'
#         fig.update_yaxes(categoryorder='total ascending')

#         # Show the plot
#         fig.show()

#     def static_barchart_exclamation(df):
#         # Filter the DataFrame for the year 2022

#         # if count = 0 then delete
#         df = df[df['exclamation_count'] > 0]

#         # Sort the DataFrame by 'exclamation_count' in descending order
#         df = df.sort_values(by='exclamation_count', ascending=False)

#         # Keep only the top 10 results
#         df = df.head(10)

#         # Create a horizontal bar chart with a predefined color sequence
#         fig = px.bar(
#             df,
#             x='exclamation_count',      # Set 'count' as the x-axis
#             y='n2',         # Set 'n2' as the y-axis
#             color='n2',     # Color bars by 'n2'
#             color_discrete_sequence=px.colors.sequential.Viridis,  # Use the Viridis color sequence
#             orientation='h',  # Make the bars horizontal
#             height=1200,    # Set height of the plot
#             width=900,      # Set width of the plot
#             title='The total exclamation points used per n2 from 2017 to 2021'  # Add title
#         )

#         # Apply a logarithmic scale to the x-axis
#         fig.update_xaxes(type='log')

#         # Ensure y-axis categories are sorted by 'count'
#         fig.update_yaxes(categoryorder='total ascending')

#         # Show the plot
#         fig.show()

    # Example usage with your filtered DataFrame
    # static_barchart(happy_days_overall_n2)
    # static_barchart_exclamation(happy_days_overall_n2)
    # static_barchart_n3(happy_days_overall)
    
    def static_barchart(df):
        # Filter the DataFrame for the year 2022

        # if count = 0 then delete
        df = df[df['count'] > 0]

        # Sort the DataFrame by 'count' in descending order
        df = df.sort_values(by='count', ascending=False)

        # Create a horizontal bar chart with a predefined color sequence
        fig = px.bar(
            df,
            x='count',      # Set 'count' as the x-axis
            y='n2',         # Set 'n2' as the y-axis
            color='n2',     # Color bars by 'n2'
            color_discrete_sequence=px.colors.sequential.Viridis,  # Use the Viridis color sequence
            orientation='h',  # Make the bars horizontal
            height=1200,    # Set height of the plot
            width=900,      # Set width of the plot
            title='The total amount of swear words used per n2 from 2017 to 2021'  # Add title
        )

        # Ensure y-axis categories are sorted by 'count'
        fig.update_yaxes(categoryorder='total ascending')

        # Show the plot
        fig.show()

    def static_barchart_n3(df):
        # Filter the DataFrame for the year 2022

        # if count = 0 then delete
        df = df[df['count'] > 0]

        # Sort the DataFrame by 'count' in descending order
        df = df.sort_values(by='count', ascending=False)

        df = df.head(15)

        # Create a horizontal bar chart with a predefined color sequence
        fig = px.bar(
            df,
            x='count',      # Set 'count' as the x-axis
            y='n3',         # Set 'n2' as the y-axis
            color='n3',     # Color bars by 'n2'
            color_discrete_sequence=px.colors.sequential.Viridis,  # Use the Viridis color sequence
            orientation='h',  # Make the bars horizontal
            height=1200,    # Set height of the plot
            width=900,      # Set width of the plot
            title='The total amount of swear words used per n3 from 2017 to 2021'  # Add title
        )

        # Ensure y-axis categories are sorted by 'count'
        fig.update_yaxes(categoryorder='total ascending')

        # Show the plot
        fig.show()

    def static_barchart_exclamation(df):
        # Filter the DataFrame for the year 2022

        # if count = 0 then delete
        df = df[df['exclamation_count'] > 0]

        # Sort the DataFrame by 'count' in descending order
        df = df.sort_values(by='exclamation_count', ascending=False)

        # Create a horizontal bar chart with a predefined color sequence
        fig = px.bar(
            df,
            x='exclamation_count',      # Set 'count' as the x-axis
            y='n2',         # Set 'n2' as the y-axis
            color='n2',     # Color bars by 'n2'
            color_discrete_sequence=px.colors.sequential.Viridis,  # Use the Viridis color sequence
            orientation='h',  # Make the bars horizontal
            height=1200,    # Set height of the plot
            width=900,      # Set width of the plot
            title='The Total Explanation Points used per n2 from 2017 to 2021'  # Add title
        )

        # Apply a logarithmic scale to the x-axis
        fig.update_xaxes(type='log')

        # Ensure y-axis categories are sorted by 'count'
        fig.update_yaxes(categoryorder='total ascending')

        # Show the plot
        fig.show()

    def static_barchart_exclamation_n3(df):
        # Filter the DataFrame for the year 2022

        # if count = 0 then delete
        df = df[df['exclamation_count'] > 0]

        # Sort the DataFrame by 'count' in descending order
        df = df.sort_values(by='exclamation_count', ascending=False)

        df = df.head(15)

        # Create a horizontal bar chart with a predefined color sequence
        fig = px.bar(
            df,
            x='exclamation_count',      # Set 'count' as the x-axis
            y='n3',         # Set 'n2' as the y-axis
            color='n3',     # Color bars by 'n2'
            color_discrete_sequence=px.colors.sequential.Viridis,  # Use the Viridis color sequence
            orientation='h',  # Make the bars horizontal
            height=900,    # Set height of the plot
            width=800,      # Set width of the plot
            title='The Total Explanation Points used per n3 from 2017 to 2021'  # Add title
        )

        # Apply a logarithmic scale to the x-axis
        fig.update_xaxes(type='log')

        # Ensure y-axis categories are sorted by 'count'
        fig.update_yaxes(categoryorder='total ascending')

        # Show the plot
        fig.show()

    # Example usage with your filtered_df
    static_barchart(happy_days_overall_n2)
    static_barchart_exclamation(happy_days_overall_n2)
    static_barchart_n3(happy_days_overall)
    static_barchart_exclamation_n3(happy_days_overall)
    

    def middle_finger_barchart(df):
        
        import plotly.graph_objects as go
        # Filter the DataFrame to only include relevant categories
        df = df[df['count'] > 0]

        # Manually define the order of categories to form the middle finger shape
        # finger_order = ['74', '45', '47', '43', '82']
        finger_order = ['69', '72', '47', '74', '82']

        # Assuming you have mapped the actual 'n2' values to these descriptive labels
        df['n2'] = pd.Categorical(df['n2'], categories=finger_order, ordered=True)

        # Sort the DataFrame by this manual order
        df = df.sort_values(by='n2')

        # Create the figure
        fig = go.Figure()

        # Add the bars (hand)
        fig.add_trace(go.Bar(
            x=df['n2'],
            y=df['count'] + 10,  # Extend bars to start from -10
            marker_color='sandybrown',  # Hand-like color for the bars
            width=0.9  # Width of the bars
        ))

        # Add a knuckle (rectangle) at the top of the bar where n2 == '47'
        for i, n2 in enumerate(df['n2']):
            if n2 == '47':
                fig.add_shape(type='rect',
                              x0=i - 0.15, y0=df['count'].iloc[i] + 10, 
                              x1=i + 0.15, y1=df['count'].iloc[i] + 12,
                              line=dict(color="black", width=20),
                              fillcolor="black")
            else:
                # Add the first horizontal line closer to the top of the bars and make it thinner
                fig.add_shape(type='line',
                              x0=i - 0.15, y0=df['count'].iloc[i] + 9, 
                              x1=i + 0.15, y1=df['count'].iloc[i] + 9,
                              line=dict(color="black", width=1.5))  # Thinner line

                # Add the second horizontal line slightly below the first and make it thinner
                fig.add_shape(type='line',
                              x0=i - 0.15, y0=df['count'].iloc[i] + 7.5, 
                              x1=i + 0.15, y1=df['count'].iloc[i] + 7.5,
                              line=dict(color="black", width=1.5))  # Thinner line

        # Update layout
        fig.update_layout(
            yaxis=dict(range=[-10, df['count'].max() + 20]),  # Extend y-axis to make space for knuckles
            title='Is the distribution of swear words across industries trying to tell us something? 🤷‍♂️',
            height=800,
            width=1000,
            showlegend=False  # Hide legend if you don't want to show it
        )

        # Show the plot
        fig.show()

    # Example usage with your filtered_df
    middle_finger_barchart(happy_days_overall_n2)
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Apply log transformation
    happy_days['log_count'] = np.log1p(happy_days['count'])  # np.log1p is used to avoid log(0) issues
    happy_days['log_exclamation_count'] = np.log1p(happy_days['exclamation_count'])

    # Scatter plot with log transformation
    plt.figure(figsize=(10, 6))
    sns.regplot(x='log_count', y='log_exclamation_count', data=happy_days, scatter_kws={'s':10}, line_kws={'color':'red'})
    plt.title('Scatter Plot with Log Transformation')
    plt.xlabel('Log of Count')
    plt.ylabel('Log of Exclamation Count')
    plt.show()
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from ipywidgets import interact, Dropdown

    def plot_n3_2(df):   
        def plot_n3_by_n3(n3):
            # Filter the DataFrame based on selected 'n3'
            filtered_df = df[df['n3'] == n3]

            # Sort by 'aar'
            filtered_df = filtered_df.sort_values(by='aar')

            # Convert 'aar' to string to ensure it's treated as a categorical variable
            filtered_df['aar'] = filtered_df['aar'].astype(str)

            # Create a new DataFrame with the selected columns
            filtered_df = filtered_df[['aar', 'count', 'exclamation_count']]

            # Create a subplot with 2 y-axes
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Add trace for 'count' on the first y-axis
            fig.add_trace(
                go.Scatter(
                    x=filtered_df['aar'], 
                    y=filtered_df['count'], 
                    name='Swear Words',
                    mode='lines+markers',
                    line=dict(color='darkgreen')  # Set line color to dark green
                ),
                secondary_y=False,
            )

            # Add trace for 'exclamation_count' on the second y-axis
            fig.add_trace(
                go.Scatter(
                    x=filtered_df['aar'], 
                    y=filtered_df['exclamation_count'], 
                    name='!!!!!!',
                    mode='lines+markers', 
                    line=dict(color='#66CDAA', dash='dash')  # Set line color to light green with dash
                ),
                secondary_y=True,
            )

            # Add titles and labels
            fig.update_layout(
                title_text=f"Swear Words & Explanation Points time series for n3={n3}",
                template='presentation',
                height=650,
                width=975,
            )

            # Set x-axis title
            fig.update_xaxes(title_text="Year", type='category')  # Ensuring categorical x-axis

            # Set y-axes titles
            fig.update_yaxes(title_text="<b>Swear Words</b>", secondary_y=False)
            fig.update_yaxes(title_text="<b>Explanation Points</b>", secondary_y=True)

            # Show the plot
            fig.show()

        # Dropdown menu for selecting the 'n3' value
        n3_selector = Dropdown(options=sorted(df['n3'].unique()), description='Select n3:')

        # Interactive widget setup
        interact(plot_n3_by_n3, n3=n3_selector)

    # Example usage with your DataFrame 'df'
    plot_n3_2(happy_days_by_year)
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from ipywidgets import interact, Dropdown

    def plot_n2_2(df):   
        def plot_n2_by_n2(n2):
            # Filter the DataFrame based on selected 'n2'
            filtered_df = df[df['n2'] == n2]

            # Sort by 'aar'
            filtered_df = filtered_df.sort_values(by='aar')

            # Convert 'aar' to string to ensure it's treated as a categorical variable
            filtered_df['aar'] = filtered_df['aar'].astype(str)

            # Create a new DataFrame with the selected columns
            filtered_df = filtered_df[['aar', 'count', 'exclamation_count']]

            # Create a subplot with 2 y-axes
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Add trace for 'count' on the first y-axis
            fig.add_trace(
                go.Scatter(
                    x=filtered_df['aar'], 
                    y=filtered_df['count'], 
                    name='Swear Words',
                    mode='lines+markers',
                    line=dict(color='darkgreen')  # Set line color to dark green
                ),
                secondary_y=False,
            )

            # Add trace for 'exclamation_count' on the second y-axis
            fig.add_trace(
                go.Scatter(
                    x=filtered_df['aar'], 
                    y=filtered_df['exclamation_count'], 
                    name='!!!!!!',
                    mode='lines+markers', 
                    line=dict(color='#66CDAA', dash='dash')  # Set line color to light green with dash
                ),
                secondary_y=True,
            )

            # Add titles and labels
            fig.update_layout(
                title_text=f"Swear Words & Explanation Points time series for n2={n2}",
                template='presentation',
                height=650,
                width=975,
            )

            # Set x-axis title
            fig.update_xaxes(title_text="Year", type='category')  # Ensuring categorical x-axis

            # Set y-axes titles
            fig.update_yaxes(title_text="<b>Swear Words</b>", secondary_y=False)
            fig.update_yaxes(title_text="<b>Explanation Points</b>", secondary_y=True)

            # Show the plot
            fig.show()

        # Dropdown menu for selecting the 'n2' value
        n2_selector = Dropdown(options=sorted(df['n2'].unique()), description='Select n2:')

        # Interactive widget setup
        interact(plot_n2_by_n2, n2=n2_selector)

    # Example usage with your DataFrame 'df'
    plot_n2_2(happy_days_by_year_n2)
    
    def remove_outliers_std_industry(df, column, std_dev=3):
        """
        Removes outliers from a DataFrame based on a specified number of standard deviations from the mean.

        Parameters:
        - df: pandas DataFrame
        - column: the column name for which to remove outliers
        - std_dev: the number of standard deviations to use (industry standard is typically 3)

        Returns:
        - df_filtered: DataFrame with outliers removed
        """
        # Calculate the mean and standard deviation of the column
        mean = df[column].mean()
        std = df[column].std()

        # Define the upper and lower bounds
        lower_bound = mean - std_dev * std
        upper_bound = mean + std_dev * std

        # Filter the DataFrame to remove outliers
        df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

        return df_filtered

    # Example usage:
    everything_cleaned = remove_outliers_std_industry(everything, 'exclamation_count', std_dev=2)
    
    def plot_aggregated_data(df):
        # Sort by 'aar'
        df = df.sort_values(by='aar')

        # Convert 'aar' to string to ensure it's treated as a categorical variable
        df['aar'] = df['aar'].astype(str)

        # Create a subplot with 2 y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add trace for 'count' on the first y-axis
        fig.add_trace(
            go.Scatter(
                x=df['aar'], 
                y=df['count'], 
                name='Swear Words',
                mode='lines+markers',
                line=dict(color='darkgreen')  # Set line color to dark green
            ),
            secondary_y=False,
        )

        # Add trace for 'exclamation_count' on the second y-axis
        fig.add_trace(
            go.Scatter(
                x=df['aar'], 
                y=df['exclamation_count'], 
                name='Explanation Points <!>',
                mode='lines+markers', 
                line=dict(color='#66CDAA', dash='dash')  # Set line color to light green with dash
            ),
            secondary_y=True,
        )

        # Add titles and labels
        fig.update_layout(
            title_text="Swear Words & Explanation Points <!>",
            template='presentation',
            height=650,
            width=975,
        )

        # Set x-axis title
        fig.update_xaxes(title_text="Year", type='category')  # Ensuring categorical x-axis

        # Set y-axes titles and ensure 0 is visible
        fig.update_yaxes(title_text="<b>Swear Words</b>", secondary_y=False, range=[0, df['count'].max() * 1.1])
        fig.update_yaxes(title_text="<b>Explanation Points</b>", secondary_y=True, range=[0, df['exclamation_count'].max() * 1.1])

        # Show the plot
        fig.show()

    # Example usage with your DataFrame 'everything'
    plot_aggregated_data(everything_cleaned)
    # plot_aggregated_data(everything)






    
