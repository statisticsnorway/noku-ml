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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
import geopandas as gpd
from scikeras.wrappers import KerasRegressor
import tensorflow as tf
import matplotlib.pyplot as plt
import dapla as dp
import datetime
from dapla.auth import AuthClient
from dapla import FileClient
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import requests
import shap


import sys

sys.path.append("../functions")
import kommune_pop
import kommune_inntekt
import kpi
import ao
import kommune_translate

fs = FileClient.get_gcs_file_system()
import numpy as np


import warnings

warnings.filterwarnings("ignore")

import math

# good_df = ao.rette_bedrifter(good_df)

import input_data
import create_datafiles

from joblib import Parallel, delayed
import multiprocessing

import time

def hente_training_data(year):
    
    fil_path = [
        f
        for f in fs.glob(
            f"gs://ssb-strukt-naering-data-produkt-prod/naringer/inndata/maskin-laering/temp/training_data.parquet"
        )
        if f.endswith(".parquet")
    ]

    # Use the ParquetDataset to read multiple files
    dataset = pq.ParquetDataset(fil_path, filesystem=fs)
    table = dataset.read()

    # Convert to Pandas DataFrame
    training_data = table.to_pandas()
        
    fil_path = [
        f
        for f in fs.glob(
            f"gs://ssb-strukt-naering-data-produkt-prod/naringer/inndata/maskin-laering/temp/imputatable_df.parquet"
        )
        if f.endswith(".parquet")
    ]

    # Use the ParquetDataset to read multiple files
    dataset = pq.ParquetDataset(fil_path, filesystem=fs)
    table = dataset.read()

    # Convert to Pandas DataFrame
    imputatable_df = table.to_pandas()
    
    fil_path = [
        f
        for f in fs.glob(
            f"gs://ssb-strukt-naering-data-produkt-prod/naringer/klargjorte-data/statistikkfiler/aar=2021/statistikkfil_foretak_pub.parquet"
        )
        if f.endswith(".parquet")
    ]

    # Use the ParquetDataset to read multiple files
    dataset = pq.ParquetDataset(fil_path, filesystem=fs)
    table = dataset.read()

    # Convert to Pandas DataFrame
    foretak_pub = table.to_pandas()
    
    foretak_pub['n3'] = foretak_pub['naring_f'].str[:4]
    foretak_pub['n2'] = foretak_pub['naring_f'].str[:2]
    
    foretak_pub = foretak_pub[foretak_pub['n2'].isin(['45', '46', '47'])]
    foretak_pub = foretak_pub[['n3',
                             'bearbeidingsverdi',
                             'produktinnsats',
                             'produksjonsverdi',
                             'omsetning',
                             'sysselsetting_syss',
                             'ts_forbruk',
                            'ts_avanse',
                            'ts_salgsint',
                            'ts_vikarutgifter',
                            'ts_byggvirk',
                            'ts_varehan',
                            'ts_anlegg',
                            'ts_tjeneste',
                            'ts_industri',
                            'ts_agentur',
                            'ts_detalj',
                            'ts_engros',
                            'ts_internet_salg',
                            'ts_annet',
                            'nopost_lonnskostnader',
                            'nopost_driftskostnader',
                            'nopost_driftsresultat',
                            'nopost_driftsinntekter',
                            'saldo_kjop_p0580']]
        
    return training_data, imputatable_df, foretak_pub

def xgboost_model(training_df, scaler, df_estimeres, year, GridSearch=True):
    """
    Trains an XGBoost model for predicting new_oms values with an optional GridSearch for hyperparameter tuning.

    Parameters:
    training_df (pd.DataFrame): DataFrame containing the training data.
    scaler (object): Scaler object for numerical features (e.g., StandardScaler, RobustScaler).
    df_estimeres (pd.DataFrame): DataFrame containing the data to be imputed.
    GridSearch (bool): Whether to perform GridSearch for hyperparameter tuning. Default is True.

    Returns:
    pd.DataFrame: DataFrame with predicted new_oms values.
    """
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    import matplotlib.pyplot as plt
    import shap
    
    print('starting ml modell. Cleaning data')

    # Make copies of the input DataFrames
    df = training_df.copy()
    imputed_df = df_estimeres.copy()
    
    # Drop rows with NaN values in the target column
    df = df.dropna(subset=['new_oms'])
    
    # Convert specified columns to category type
    categorical_columns = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    for col in categorical_columns:
        df[col] = df[col].astype("category")

    # Define features and target variable
    X = df.drop(columns=["new_oms"])
    y = df["new_oms"]

    # Define categorical and numerical features
    # categorical_features = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    # numerical_features = [
    #     "inntekt_delta_oms",
    #     "emp_delta_oms",
    #     "befolkning_delta_oms",
    #     "inflation_rate_oms",
    #     "gjeldende_bdr_syss",
    #     "new_oms_trendForecast", 
    #     'oms_syssmean_basedOn_naring',
    #     'oms_syssmean_basedOn_naring_kommune'
    # ]
    
    categorical_features = ["nacef_5", "b_kommunenr"]
    numerical_features = [
        "new_oms_trendForecast", 
        'oms_syssmean_basedOn_naring_kommune'
    ]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, numerical_features),  # Apply scaling to numerical features
            ("cat", OneHotEncoder(categories="auto", handle_unknown="ignore"), categorical_features),  # One-hot encoding for categorical features
        ]
    )

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the preprocessor on the training data
    preprocessor.fit(X_train)

    # Transform the training and testing data
    X_train_transformed = preprocessor.transform(X_train).toarray()
    X_test_transformed = preprocessor.transform(X_test).toarray()

    if GridSearch:
        # Define the model
        regressor = xgb.XGBRegressor(eval_metric="rmse", random_state=42)

        # Define parameter grid for GridSearch
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        # Perform GridSearch with cross-validation
        grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)
        grid_search.fit(X_train_transformed, y_train)

        # Print best parameters
        print("Best parameters found by GridSearch:", grid_search.best_params_)

        # Use best estimator from grid search
        regressor = grid_search.best_estimator_
    else:
        # Define the model with default parameters
        regressor = xgb.XGBRegressor(eval_metric="rmse", random_state=42)

        # Train the model
        eval_set = [(X_train_transformed, y_train), (X_test_transformed, y_test)]
        regressor.fit(X_train_transformed, y_train, eval_set=eval_set, verbose=False)

    # Evaluate the model
    y_pred = regressor.predict(X_test_transformed)

    # Check for negative values in predictions
    negative_indices = np.where(y_pred < 0)[0]
    negative_predictions = y_pred[y_pred < 0]

    if len(negative_predictions) > 0:
        print("Number of negative predictions:", len(negative_predictions))
    else:
        print("No negative predictions found.")
        
    # Set negative predictions to zero
    y_pred = np.maximum(y_pred, 0)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)  # Calculate Mean Absolute Error
    print("Mean Squared Error:", mse)
    print("R-squared:", r_squared)
    print("Mean Absolute Error:", mae)
    
    X_test['n3'] = X_test['nacef_5'].str[:4]
    
    # Evaluate performance based on the n3 class
    results = pd.DataFrame({'n3': X_test['n3'], 'actual': y_test, 'predicted': y_pred})
    # Define the n3 categories to exclude
    n3_to_exclude = ['45.1', '45.2', '46.3', '46.4', '46.5', '46.7', '46.9', '10.4', '02.4']

    # Check if there are any n3 categories not in the excluded list
    if not results['n3'].isin(n3_to_exclude).all():
        # Filter out the rows where the n3 is in the excluded list
        filtered_results = results[~results['n3'].isin(n3_to_exclude)]

        # Extract the actual and predicted values after filtering
        filtered_y_test = filtered_results['actual']
        filtered_y_pred = filtered_results['predicted']

        # Recalculate the evaluation metrics excluding the specified n3 categories
        filtered_mse = mean_squared_error(filtered_y_test, filtered_y_pred)
        filtered_mae = mean_absolute_error(filtered_y_test, filtered_y_pred)
        filtered_medae = median_absolute_error(filtered_y_test, filtered_y_pred)
        filtered_r_squared = r2_score(filtered_y_test, filtered_y_pred)
        filtered_rmse = np.sqrt(filtered_mse)

        # Print out the filtered metrics
        print(f"Filtered Mean Squared Error (MSE): {filtered_mse}")
        print(f"Filtered Mean Absolute Error (MAE): {filtered_mae}")
        print(f"Filtered Median Absolute Error (MedAE): {filtered_medae}")
        print(f"Filtered R-squared score: {filtered_r_squared}")
        print(f"Filtered Root Mean Squared Error (RMSE): {filtered_rmse}")
    else:
        print("No valid n3 categories found after exclusion. Skipping filtered metrics calculation.")
    
    metrics_per_n3 = results.groupby('n3').apply(lambda group: pd.Series({
        'mse': mean_squared_error(group['actual'], group['predicted']),
        'r_squared': r2_score(group['actual'], group['predicted']),
        'mae': mean_absolute_error(group['actual'], group['predicted'])
    })).reset_index()
    
    print("Metrics per 'n3':")
    print(metrics_per_n3)

    # Plot the learning history
    results = regressor.evals_result()
    epochs = len(results["validation_0"]["rmse"])
    x_axis = range(0, epochs)
    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, results["validation_0"]["rmse"], label="Train")
    plt.plot(x_axis, results["validation_1"]["rmse"], label="Test")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.title("XGBoost Learning History")
    plt.show()

    # Plot Predicted vs. Actual Values
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs. Actual Values")
    plt.show()

    # Plot Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, residuals, alpha=0.3)
    plt.hlines(0, y_test.min(), y_test.max(), colors="r", linestyles="dashed")
    plt.xlabel("Actual")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")
    plt.show()
    

    # Tree textual representation
    booster = regressor.get_booster()
    with open("dump.raw.txt", "w") as f:
        f.write("\n".join(booster.get_dump()))
    print(booster.get_dump()[0])  # Print the first tree

    # SHAP values
    explainer = shap.TreeExplainer(regressor, X_train_transformed)
    shap_values = explainer.shap_values(X_test_transformed)

    # Get feature names after one-hot encoding
    feature_names = preprocessor.get_feature_names_out()

    # Summary plot of SHAP values
    shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names)

    # Force plot for a single prediction (e.g., the first instance)
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[0], X_test_transformed[0], feature_names=feature_names)

    # Find the correct index for the feature "verdi"
    verdi_index = list(feature_names).index("num__new_oms_trendForecast")

    # Dependence plot to show the effect of a single feature across the dataset
    shap.dependence_plot(verdi_index, shap_values, X_test_transformed, feature_names=feature_names)

    # Impute the missing data
    imputed_X = imputed_df.drop(columns=["new_oms"])
    imputed_X_transformed = preprocessor.transform(imputed_X)
    imputed_df["predicted_oms"] = regressor.predict(imputed_X_transformed)

    # Ensure no negative predictions
    imputed_df['predicted_oms'] = imputed_df['predicted_oms'].clip(lower=0)
    imputed_df['predicted_oms'] = imputed_df['predicted_oms'].astype(float)
    
    return imputed_df

def xgboost_model_with_pca(training_df, scaler, df_estimeres, year, GridSearch=True, apply_pca=True, n_components=6):
    """
    Trains an XGBoost model for predicting new_oms values with optional PCA for dimensionality reduction
    and GridSearch for hyperparameter tuning. Includes visualizations for explained variance, learning history,
    and SHAP values.
    
    Parameters:
    training_df (pd.DataFrame): DataFrame containing the training data.
    scaler (object): Scaler object for numerical features (e.g., StandardScaler, RobustScaler).
    df_estimeres (pd.DataFrame): DataFrame containing the data to be imputed.
    GridSearch (bool): Whether to perform GridSearch for hyperparameter tuning. Default is True.
    apply_pca (bool): Whether to apply PCA for dimensionality reduction. Default is True.
    n_components (int, float, or None): Number of components to keep after applying PCA. If None, it will not reduce dimensions.
    
    Returns:
    pd.DataFrame: DataFrame with predicted new_oms values.
    """
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import pandas as pd
    import shap
    import plotly.graph_objects as go
    
    print('Starting the XGBoost model with PCA...')

    # Make copies of the input DataFrames
    df = training_df.copy()
    imputed_df = df_estimeres.copy()
    
    # Drop rows with NaN values in the target column
    df = df.dropna(subset=['new_oms'])
    
    # Convert specified columns to category type
    categorical_columns = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    for col in categorical_columns:
        df[col] = df[col].astype("category")

    # Define features and target variable
    X = df.drop(columns=["new_oms"])
    y = df["new_oms"]

    # Define categorical and numerical features
    categorical_features = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    numerical_features = [
        "inntekt_delta_oms",
        "emp_delta_oms",
        "befolkning_delta_oms",
        "inflation_rate_oms",
        "gjeldende_bdr_syss",
        "new_oms_trendForecast", 
        'oms_syssmean_basedOn_naring',
        'oms_syssmean_basedOn_naring_kommune'
    ]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, numerical_features),  # Apply scaling to numerical features
            ("cat", OneHotEncoder(categories="auto", handle_unknown="ignore"), categorical_features),  # One-hot encoding for categorical features
        ]
    )

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the preprocessor on the training data
    preprocessor.fit(X_train)

    # Transform the training and testing data
    X_train_transformed = preprocessor.transform(X_train).toarray()
    X_test_transformed = preprocessor.transform(X_test).toarray()

    # Apply PCA if requested
    if apply_pca:
        pca = PCA(n_components=n_components)
        X_train_transformed = pca.fit_transform(X_train_transformed)
        X_test_transformed = pca.transform(X_test_transformed)

        # Visualize explained variance
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(explained_variance) + 1),
            y=explained_variance,
            mode='lines+markers',
            marker=dict(size=10, color='lightgreen'),
            line=dict(color='lightgreen', width=3),
            hovertemplate='Component %{x}<br>Cumulative Explained Variance: %{y:.2f}<extra></extra>'
        ))
        fig.update_layout(
            title='Cumulative Explained Variance by PCA Components',
            xaxis_title='Number of Components',
            yaxis_title='Cumulative Explained Variance',
            template="plotly_white",
            font=dict(size=14)
        )
        fig.show()

    # Define the model and perform GridSearch if requested
    if GridSearch:
        regressor = xgb.XGBRegressor(eval_metric="rmse", random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)
        grid_search.fit(X_train_transformed, y_train)
        print("Best parameters found by GridSearch:", grid_search.best_params_)
        regressor = grid_search.best_estimator_
    else:
        regressor = xgb.XGBRegressor(eval_metric="rmse", random_state=42)
        eval_set = [(X_train_transformed, y_train), (X_test_transformed, y_test)]
        regressor.fit(X_train_transformed, y_train, eval_set=eval_set, verbose=False)

    # Evaluate the model
    y_pred = regressor.predict(X_test_transformed)
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("R-squared:", r_squared)
    print("Mean Absolute Error:", mae)

    # Plot the learning history
    results = regressor.evals_result()
    epochs = len(results["validation_0"]["rmse"])
    x_axis = range(0, epochs)
    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, results["validation_0"]["rmse"], label="Train")
    plt.plot(x_axis, results["validation_1"]["rmse"], label="Test")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.title("XGBoost Learning History")
    plt.show()

    # Plot Predicted vs. Actual Values
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs. Actual Values")
    plt.show()

    # Plot Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, residuals, alpha=0.3)
    plt.hlines(0, y_test.min(), y_test.max(), colors="r", linestyles="dashed")
    plt.xlabel("Actual")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")
    plt.show()

    # SHAP values if PCA was not applied
    if not apply_pca:
        explainer = shap.TreeExplainer(regressor, X_train_transformed)
        shap_values = explainer.shap_values(X_test_transformed)
        feature_names = preprocessor.get_feature_names_out()
        shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names)

    # Impute the missing data
    imputed_X = imputed_df.drop(columns=["new_oms"])
    imputed_X_transformed = preprocessor.transform(imputed_X)

    if apply_pca:
        imputed_X_transformed = pca.transform(imputed_X_transformed)

    imputed_df["predicted_oms"] = regressor.predict(imputed_X_transformed)
    imputed_df['predicted_oms'] = imputed_df['predicted_oms'].clip(lower=0).astype(float)
    
    return imputed_df




def knn_model(training_df, scaler, df_estimeres, year, GridSearch=True):
    """
    Trains a K-Nearest Neighbors model for predicting new_oms values with an optional GridSearch for hyperparameter tuning.

    Parameters:
    training_df (pd.DataFrame): DataFrame containing the training data.
    scaler (object): Scaler object for numerical features (e.g., StandardScaler, RobustScaler).
    df_estimeres (pd.DataFrame): DataFrame containing the data to be imputed.
    GridSearch (bool): Whether to perform GridSearch for hyperparameter tuning. Default is True.

    Returns:
    pd.DataFrame: DataFrame with predicted new_oms values.
    """
    import numpy as np
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.neighbors import KNeighborsRegressor
    import matplotlib.pyplot as plt
    import pandas as pd

    # Make copies of the input DataFrames
    df = training_df.copy()
    imputed_df = df_estimeres.copy()
    
    categorical_columns = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    df[categorical_columns] = df[categorical_columns].astype(str)
    imputed_df[categorical_columns] = imputed_df[categorical_columns].astype(str)

    # Columns to fill with 'missing' and 0 respectively
    columns_to_fill = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    numeric_columns_to_fill = [
        "inntekt_delta_oms",
        "emp_delta_oms",
        "befolkning_delta_oms",
        "inflation_rate_oms",
        "gjeldende_bdr_syss",
        "new_oms_trendForecast",
        'oms_syssmean_basedOn_naring',
        'oms_syssmean_basedOn_naring_kommune'
    ]

    # Fill NaN values with 'missing' for the specified columns
    df[columns_to_fill] = df[columns_to_fill].fillna('missing')
    imputed_df[columns_to_fill] = imputed_df[columns_to_fill].fillna('missing')
    
    # Fill NaN values with 0 for the specified columns
    df[numeric_columns_to_fill] = df[numeric_columns_to_fill].fillna(0)
    imputed_df[numeric_columns_to_fill] = imputed_df[numeric_columns_to_fill].fillna(0)

    # Convert specified columns to category type
    # categorical_columns = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    for col in categorical_columns:
        df[col] = df[col].astype("category")

    # Define features and target
    X = df.drop(columns=["new_oms"])
    y = df["new_oms"]

    # Define preprocessor
    categorical_features = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    
    numerical_features = [
        "inntekt_delta_oms",
        "emp_delta_oms",
        "befolkning_delta_oms",
        "inflation_rate_oms",
        "gjeldende_bdr_syss",
        "new_oms_trendForecast",
        'oms_syssmean_basedOn_naring',
        'oms_syssmean_basedOn_naring_kommune'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, numerical_features),
            ("cat", OneHotEncoder(categories="auto", handle_unknown="ignore"), categorical_features),
        ]
    )

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the preprocessor and transform the training and testing data
    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    if GridSearch:
        # Define the model
        regressor = KNeighborsRegressor()

        # Define parameter grid for GridSearch
        param_grid = {
            'n_neighbors': [2, 3, 5, 7]
        }

        # Perform GridSearch with cross-validation
        grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)
        grid_search.fit(X_train_transformed, y_train)

        # Print best parameters
        print("Best parameters found by GridSearch:", grid_search.best_params_)

        # Use best estimator from grid search
        regressor = grid_search.best_estimator_
    else:
        # Define the model with default parameters
        regressor = KNeighborsRegressor(n_neighbors=2)

        # Train the model
        regressor.fit(X_train_transformed, y_train)

    # Perform cross-validation using MAE as the scoring metric
    cv_scores = cross_val_score(regressor, X_train_transformed, y_train, cv=5, scoring='neg_mean_absolute_error')

    # Since cross_val_score returns negative values for error metrics, we negate them to get the actual MAE
    mean_mae = -np.mean(cv_scores)
    std_mae = np.std(cv_scores)

    print(f"Cross-Validated Mean MAE: {mean_mae}")
    print(f"Cross-Validated MAE Standard Deviation: {std_mae}")


    # Predict on test data
    y_pred = regressor.predict(X_test_transformed)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("R-squared:", r_squared)
    print("Mean Absolute Error:", mae)
    
    # Calculate MAE per year and print it
    results = X_test.copy()
    results['actual'] = y_test
    results['predicted'] = y_pred
    if 'year' in results.columns:
        mae_per_year = results.groupby('year').apply(lambda group: mean_absolute_error(group['actual'], group['predicted']))
        print("\nMean Absolute Error per Year:")
        print(mae_per_year)
    
    # Create the n3 class by taking the first 4 characters of nacef_5
    X_test['n3'] = X_test['nacef_5'].str[:4]
    
    # Evaluate performance based on the n3 class
    results = pd.DataFrame({'n3': X_test['n3'], 'actual': y_test, 'predicted': y_pred})
    
    # Define the n3 categories to exclude
    n3_to_exclude = ['45.1', '45.2', '46.3', '46.4', '46.5', '46.7', '46.9', '10.4', '02.4']

    # Check if there are any n3 categories not in the excluded list
    if not results['n3'].isin(n3_to_exclude).all():
        # Filter out the rows where the n3 is in the excluded list
        filtered_results = results[~results['n3'].isin(n3_to_exclude)]

        # Extract the actual and predicted values after filtering
        filtered_y_test = filtered_results['actual']
        filtered_y_pred = filtered_results['predicted']

        # Recalculate the evaluation metrics excluding the specified n3 categories
        filtered_mse = mean_squared_error(filtered_y_test, filtered_y_pred)
        filtered_mae = mean_absolute_error(filtered_y_test, filtered_y_pred)
        filtered_r_squared = r2_score(filtered_y_test, filtered_y_pred)
        filtered_rmse = np.sqrt(filtered_mse)

        # Print out the filtered metrics
        print(f"Filtered Mean Squared Error (MSE): {filtered_mse}")
        print(f"Filtered Mean Absolute Error (MAE): {filtered_mae}")
        print(f"Filtered R-squared score: {filtered_r_squared}")
        print(f"Filtered Root Mean Squared Error (RMSE): {filtered_rmse}")
    else:
        print("No valid n3 categories found after exclusion. Skipping filtered metrics calculation.")

    metrics_per_n3 = results.groupby('n3').apply(lambda group: pd.Series({
        'mse': mean_squared_error(group['actual'], group['predicted']),
        'r_squared': r2_score(group['actual'], group['predicted']),
        'mae': mean_absolute_error(group['actual'], group['predicted'])
    })).reset_index()
    
    print("Metrics per 'n3':")
    print(metrics_per_n3)
    
    # Plot Predicted vs. Actual Values
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs. Actual Values")
    plt.show()

    # Plot Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, residuals, alpha=0.3)
    plt.hlines(0, y_test.min(), y_test.max(), colors="r", linestyles="dashed")
    plt.xlabel("Actual")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")
    plt.show()

    # Impute the missing data
    imputed_X = imputed_df.drop(columns=["new_oms"])
    imputed_X_transformed = preprocessor.transform(imputed_X)
    imputed_df["predicted_oms"] = regressor.predict(imputed_X_transformed)
    
    return imputed_df

# def knn_model_with_pca(training_df, scaler, df_estimeres, year, GridSearch=True, apply_pca=True, n_components=5):

#     """
#     Trains a K-Nearest Neighbors model for predicting new_oms values with optional PCA for dimensionality reduction
#     and optional GridSearch for hyperparameter tuning. Includes interactive plot for explained variance.

#     Parameters:
#     training_df (pd.DataFrame): DataFrame containing the training data.
#     scaler (object): Scaler object for numerical features (e.g., StandardScaler, RobustScaler).
#     df_estimeres (pd.DataFrame): DataFrame containing the data to be imputed.
#     GridSearch (bool): Whether to perform GridSearch for hyperparameter tuning. Default is True.
#     apply_pca (bool): Whether to apply PCA for dimensionality reduction. Default is True.
#     n_components (int, float, or None): Number of components to keep after applying PCA. If None, it will not reduce dimensions.
#                                          If a float is given (e.g., 0.95), PCA will select the number of components that explain that proportion of variance.

#     Returns:
#     pd.DataFrame: DataFrame with predicted new_oms values.
#     """
#     from sklearn.decomposition import PCA
#     import matplotlib.pyplot as plt
#     import plotly.graph_objects as go
#     import numpy as np
#     from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
#     from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
#     from sklearn.preprocessing import OneHotEncoder
#     from sklearn.compose import ColumnTransformer
#     from sklearn.neighbors import KNeighborsRegressor
#     import pandas as pd
    
#     # Make copies of the input DataFrames
#     df = training_df.copy()
#     imputed_df = df_estimeres.copy()
    
#     categorical_columns = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
#     df[categorical_columns] = df[categorical_columns].astype(str)
#     imputed_df[categorical_columns] = imputed_df[categorical_columns].astype(str)

#     columns_to_fill = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
#     numeric_columns_to_fill = [
#         "inntekt_delta_oms", "emp_delta_oms", "befolkning_delta_oms", 
#         "inflation_rate_oms", "gjeldende_bdr_syss", "new_oms_trendForecast",
#         'oms_syssmean_basedOn_naring', 'oms_syssmean_basedOn_naring_kommune'
#     ]

#     # Fill missing values
#     df[columns_to_fill] = df[columns_to_fill].fillna('missing')
#     imputed_df[columns_to_fill] = imputed_df[columns_to_fill].fillna('missing')
#     df[numeric_columns_to_fill] = df[numeric_columns_to_fill].fillna(0)
#     imputed_df[numeric_columns_to_fill] = imputed_df[numeric_columns_to_fill].fillna(0)

#     # Define features and target
#     X = df.drop(columns=["new_oms"])
#     y = df["new_oms"]

#     categorical_features = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
#     numerical_features = [
#         "inntekt_delta_oms", "emp_delta_oms", "befolkning_delta_oms", 
#         "inflation_rate_oms", "gjeldende_bdr_syss", "new_oms_trendForecast", 
#         'oms_syssmean_basedOn_naring', 'oms_syssmean_basedOn_naring_kommune'
#     ]

#     # Preprocessing pipeline
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", scaler, numerical_features),
#             ("cat", OneHotEncoder(categories="auto", handle_unknown="ignore"), categorical_features),
#         ]
#     )

#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Fit the preprocessor and transform the training and testing data
#     preprocessor.fit(X_train)
#     X_train_transformed = preprocessor.transform(X_train)
#     X_test_transformed = preprocessor.transform(X_test)

#     pca = PCA(n_components=n_components)
#     X_train_pca = pca.fit_transform(X_train_transformed)
#     X_test_pca = pca.transform(X_test_transformed)

#     # Automatically create and display the interactive PCA plot
#     explained_variance = np.cumsum(pca.explained_variance_ratio_)
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=np.arange(1, len(explained_variance) + 1),
#         y=explained_variance,
#         mode='lines+markers',
#         marker=dict(size=8),
#         hovertemplate='Component %{x}<br>Cumulative Explained Variance: %{y:.2f}<extra></extra>',
#         line=dict(dash='dash', color='blue')
#     ))

#     # Update layout
#     fig.update_layout(
#         title='Cumulative Explained Variance by PCA Components',
#         xaxis_title='Number of Components',
#         yaxis_title='Cumulative Explained Variance',
#         template="plotly_white"
#     )

#     # Show the plot
#     fig.show()
    
#     # Get the PCA components and their corresponding feature importance
#     pca_components = pd.DataFrame(
#         pca.components_,
#         columns=numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out()),
#         index=[f"PC{i+1}" for i in range(pca.n_components_)]
#     )

#     # Display the top contributing features for each component
#     for i in range(pca.n_components_):
#         print(f"\nTop features for PC{i+1}:")
#         component = pca_components.iloc[i]
#         sorted_component = component.abs().sort_values(ascending=False)
#         top_features = sorted_component.head(5).index.tolist()
#         print(f"Top contributing features: {top_features}")
#         print(component.loc[top_features])

#     X_train_transformed = X_train_pca
#     X_test_transformed = X_test_pca

#     if GridSearch:
#         # Define the model and perform GridSearch with cross-validation
#         regressor = KNeighborsRegressor()
#         param_grid = {'n_neighbors': [2, 3, 5, 7]}
#         grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)
#         grid_search.fit(X_train_transformed, y_train)
#         print("Best parameters found by GridSearch:", grid_search.best_params_)
#         regressor = grid_search.best_estimator_
#     else:
#         regressor = KNeighborsRegressor(n_neighbors=2)
#         regressor.fit(X_train_transformed, y_train)

#     # Evaluate with cross-validation
#     cv_scores = cross_val_score(regressor, X_train_transformed, y_train, cv=5, scoring='neg_mean_absolute_error')
#     mean_mae = -np.mean(cv_scores)
#     std_mae = np.std(cv_scores)
#     print(f"Cross-Validated Mean MAE: {mean_mae}")
#     print(f"Cross-Validated MAE Standard Deviation: {std_mae}")

#     # Predict on test data
#     y_pred = regressor.predict(X_test_transformed)
#     mse = mean_squared_error(y_test, y_pred)
#     r_squared = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     print("Mean Squared Error:", mse)
#     print("R-squared:", r_squared)
#     print("Mean Absolute Error:", mae)
    
#     # Calculate MAE per year and print it
#     results = X_test.copy()
#     results['actual'] = y_test
#     results['predicted'] = y_pred
    
#     if 'year' in results.columns:
#         mae_per_year = results.groupby('year').apply(lambda group: mean_absolute_error(group['actual'], group['predicted']))
#         print("\nMean Absolute Error per Year:")
#         print(mae_per_year)
   
#     # Create the n3 class by taking the first 4 characters of nacef_5
#     X_test['n3'] = X_test['nacef_5'].str[:4]
    
#     # Evaluate performance based on the n3 class
#     results = pd.DataFrame({'n3': X_test['n3'], 'actual': y_test, 'predicted': y_pred})
        
#     metrics_per_n3 = results.groupby('n3').apply(lambda group: pd.Series({
#         'mse': mean_squared_error(group['actual'], group['predicted']),
#         'r_squared': r2_score(group['actual'], group['predicted']),
#         'mae': mean_absolute_error(group['actual'], group['predicted'])
#     })).reset_index()
    
#     print("Metrics per 'n3':")
#     print(metrics_per_n3)

#     # Plot Predicted vs. Actual Values
#     plt.figure(figsize=(10, 5))
#     plt.scatter(y_test, y_pred, alpha=0.3)
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
#     plt.xlabel("Actual")
#     plt.ylabel("Predicted")
#     plt.title("Predicted vs. Actual Values")
#     plt.show()

#     # Plot Residuals
#     residuals = y_test - y_pred
#     plt.figure(figsize=(10, 5))
#     plt.scatter(y_test, residuals, alpha=0.3)
#     plt.hlines(0, y_test.min(), y_test.max(), colors="r", linestyles="dashed")
#     plt.xlabel("Actual")
#     plt.ylabel("Residuals")
#     plt.title("Residuals Plot")
#     plt.show()

#     # Impute the missing data
#     imputed_X = imputed_df.drop(columns=["new_oms"])
#     imputed_X_transformed = preprocessor.transform(imputed_X)
    
#     imputed_X_transformed = pca.transform(imputed_X_transformed)
    
#     imputed_df["predicted_oms"] = regressor.predict(imputed_X_transformed)
    
#     return imputed_df


def knn_model_with_pca(training_df, scaler, df_estimeres, year, GridSearch=False, apply_pca=True, n_components=6):
    
    """
    Trains a K-Nearest Neighbors model for predicting new_oms values with optional PCA for dimensionality reduction
    and optional GridSearch for hyperparameter tuning. Includes interactive plot for explained variance.
    
    Parameters:
    training_df (pd.DataFrame): DataFrame containing the training data.
    scaler (object): Scaler object for numerical features (e.g., StandardScaler, RobustScaler).
    df_estimeres (pd.DataFrame): DataFrame containing the data to be imputed.
    GridSearch (bool): Whether to perform GridSearch for hyperparameter tuning. Default is True.
    apply_pca (bool): Whether to apply PCA for dimensionality reduction. Default is True.
    n_components (int, float, or None): Number of components to keep after applying PCA. If None, it will not reduce dimensions.
                                         If a float is given (e.g., 0.95), PCA will select the number of components that explain that proportion of variance.
    
    Returns:
    pd.DataFrame: DataFrame with predicted new_oms values.
    """
    from sklearn.decomposition import PCA
    import plotly.express as px
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import numpy as np
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.neighbors import KNeighborsRegressor
    import pandas as pd
    
    # Make copies of the input DataFrames
    df = training_df.copy()
    imputed_df = df_estimeres.copy()
    
    def filter_previous_years(df, current_year):
        return df[df['year'] <= current_year]

    categorical_columns = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    df[categorical_columns] = df[categorical_columns].astype(str)
    imputed_df[categorical_columns] = imputed_df[categorical_columns].astype(str)

    columns_to_fill = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    numeric_columns_to_fill = [
        "inntekt_delta_oms", "emp_delta_oms", "befolkning_delta_oms", 
        "inflation_rate_oms", "gjeldende_bdr_syss", "new_oms_trendForecast",
        'oms_syssmean_basedOn_naring', 'oms_syssmean_basedOn_naring_kommune'
    ]

    # Fill missing values
    df[columns_to_fill] = df[columns_to_fill].fillna('missing')
    imputed_df[columns_to_fill] = imputed_df[columns_to_fill].fillna('missing')
    df[numeric_columns_to_fill] = df[numeric_columns_to_fill].fillna(0)
    imputed_df[numeric_columns_to_fill] = imputed_df[numeric_columns_to_fill].fillna(0)

    # Loop through each unique year, train a model for each year
    unique_years = df['year'].unique()
    
    for year in unique_years:
        print("--------------------------------")
        print(f"Training model for year: {year}")
        
        # Filter data to include only the current year and previous years
        df_filtered = filter_previous_years(df, year)
        
        # Define features and target
        X = df_filtered.drop(columns=["new_oms"])
        y = df_filtered["new_oms"]

        categorical_features = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
        numerical_features = [
            "inntekt_delta_oms", "emp_delta_oms", "befolkning_delta_oms", 
            "inflation_rate_oms", "gjeldende_bdr_syss", "new_oms_trendForecast", 
            'oms_syssmean_basedOn_naring', 'oms_syssmean_basedOn_naring_kommune'
        ]

        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", scaler, numerical_features),
                ("cat", OneHotEncoder(categories="auto", handle_unknown="ignore"), categorical_features),
            ]
        )

        # Split into training and testing sets (only train on past and current data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit the preprocessor and transform the training and testing data
        preprocessor.fit(X_train)
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        if apply_pca:
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train_transformed)
            X_test_pca = pca.transform(X_test_transformed)
            X_train_transformed = X_train_pca
            X_test_transformed = X_test_pca
      
        if GridSearch:
            # Define the model and perform GridSearch with cross-validation
            regressor = KNeighborsRegressor()
            param_grid = {'n_neighbors': [2, 3, 5, 7]}
            grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)
            grid_search.fit(X_train_transformed, y_train)
            print("Best parameters found by GridSearch:", grid_search.best_params_)
            regressor = grid_search.best_estimator_
        else:
            regressor = KNeighborsRegressor(n_neighbors=2)
            regressor.fit(X_train_transformed, y_train)

        # Predict on test data
        y_pred = regressor.predict(X_test_transformed)
        mse = mean_squared_error(y_test, y_pred)
        r_squared = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Year: {year} - Mean Squared Error:", mse)
        print(f"Year: {year} - R-squared:", r_squared)
        print(f"Year: {year} - Mean Absolute Error:", mae)
        
    # Create the n3 class by taking the first 4 characters of nacef_5
    X_test['n3'] = X_test['nacef_5'].str[:4]

    # Evaluate performance based on the n3 class
    results = pd.DataFrame({'n3': X_test['n3'], 'actual': y_test, 'predicted': y_pred})

    metrics_per_n3 = results.groupby('n3').apply(lambda group: pd.Series({
        'mse': mean_squared_error(group['actual'], group['predicted']),
        'r_squared': r2_score(group['actual'], group['predicted']),
        'mae': mean_absolute_error(group['actual'], group['predicted'])
    })).reset_index()
    
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    fig = go.Figure()

    # Create a smooth light green curve with larger markers and no dash
    fig.add_trace(go.Scatter(
        x=np.arange(1, len(explained_variance) + 1),
        y=explained_variance,
        mode='lines+markers',
        marker=dict(size=10, color='lightgreen'),  # Larger markers with light green color
        line=dict(color='lightgreen', width=3),  # Light green curve, thicker line
        hovertemplate='Component %{x}<br>Cumulative Explained Variance: %{y:.2f}<extra></extra>'
    ))

    # Update layout for cleaner presentation
    fig.update_layout(
        title='Cumulative Explained Variance by PCA Components',
        xaxis_title='Number of Components',
        yaxis_title='Cumulative Explained Variance',
        template="plotly_white",  # Use a clean white template
        font=dict(size=14)  # Adjust font size for readability
    )

    # Show the plot
    fig.show()
    
    # Get the PCA components and their corresponding feature importance
    pca_components = pd.DataFrame(
        pca.components_,
        columns=numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out()),
        index=[f"PC{i+1}" for i in range(pca.n_components_)]
    )

    # Display the top contributing features for each component
    for i in range(pca.n_components_):
        print(f"\nTop features for PC{i+1}:")
        component = pca_components.iloc[i]
        sorted_component = component.abs().sort_values(ascending=False)
        top_features = sorted_component.head(5).index.tolist()
        print(f"Top contributing features: {top_features}")
        print(component.loc[top_features])

    print("Metrics per 'n3':")
    print(metrics_per_n3)

    # Plot Predicted vs. Actual Values
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs. Actual Values. Static Plot")
    plt.show()

    # Plot Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, residuals, alpha=0.3)
    plt.hlines(0, y_test.min(), y_test.max(), colors="r", linestyles="dashed")
    plt.xlabel("Actual")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot. Static")
    plt.show()

    # Create a DataFrame to hold all information
    plot_data = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Residuals': y_test - y_pred
    })


    # Impute the missing data for the current year
    imputed_X = imputed_df.drop(columns=["new_oms"])
    imputed_X_transformed = preprocessor.transform(imputed_X)

    if apply_pca:
        imputed_X_transformed = pca.transform(imputed_X_transformed)

    imputed_df["predicted_oms"] = regressor.predict(imputed_X_transformed)
    
    return imputed_df


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV

def knn_model_with_pca_custom_distance(training_df, scaler, df_estimeres, year, GridSearch=False, apply_pca=True, n_components=6):
    """
    Trains a K-Nearest Neighbors model for predicting new_oms values with optional PCA for dimensionality reduction
    and custom distance metric to penalize future years.
    
    Parameters:
    training_df (pd.DataFrame): DataFrame containing the training data.
    scaler (object): Scaler object for numerical features (e.g., StandardScaler, RobustScaler).
    df_estimeres (pd.DataFrame): DataFrame containing the data to be imputed.
    GridSearch (bool): Whether to perform GridSearch for hyperparameter tuning. Default is True.
    apply_pca (bool): Whether to apply PCA for dimensionality reduction. Default is True.
    n_components (int, float, or None): Number of components to keep after applying PCA. If None, it will not reduce dimensions.
    
    Returns:
    pd.DataFrame: DataFrame with predicted new_oms values.
    """
    
    # Make copies of the input DataFrames
    df = training_df.copy()
    imputed_df = df_estimeres.copy()
    
    # Include year as a feature
    categorical_columns = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    df[categorical_columns] = df[categorical_columns].astype(str)
    imputed_df[categorical_columns] = imputed_df[categorical_columns].astype(str)

    columns_to_fill = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    numeric_columns_to_fill = [
        "inntekt_delta_oms", "emp_delta_oms", "befolkning_delta_oms", 
        "inflation_rate_oms", "gjeldende_bdr_syss", "new_oms_trendForecast",
        'oms_syssmean_basedOn_naring', 'oms_syssmean_basedOn_naring_kommune',
        'year'  # Add 'year' to the list of numeric columns
    ]
    
    # Fill missing values
    df[columns_to_fill] = df[columns_to_fill].fillna('missing')
    imputed_df[columns_to_fill] = imputed_df[columns_to_fill].fillna('missing')
    df[numeric_columns_to_fill] = df[numeric_columns_to_fill].fillna(0)
    imputed_df[numeric_columns_to_fill] = imputed_df[numeric_columns_to_fill].fillna(0)

    # Preprocessing pipeline
    categorical_features = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    numerical_features = [
        "inntekt_delta_oms", "emp_delta_oms", "befolkning_delta_oms", 
        "inflation_rate_oms", "gjeldende_bdr_syss", "new_oms_trendForecast", 
        'oms_syssmean_basedOn_naring', 'oms_syssmean_basedOn_naring_kommune',
        'year'  # Include 'year' in the list of features for training
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, numerical_features),
            ("cat", OneHotEncoder(categories="auto", handle_unknown="ignore"), categorical_features),
        ]
    )
    
    # Split the data into training and testing sets
    X = df.drop(columns=["new_oms"])
    y = df["new_oms"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit the preprocessor and transform the training and testing data
    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    if apply_pca:
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_transformed)
        X_test_pca = pca.transform(X_test_transformed)
        X_train_transformed = X_train_pca
        X_test_transformed = X_test_pca

    # Define a custom distance metric that penalizes future years
    def custom_distance_metric(x1, x2):
        year_diff = x1[-1] - x2[-1]  # Compare the year feature, assuming it's the last column
        if year_diff > 0:
            # Penalize if the current point is from a future year
            penalty = 1e10  # A very large number to ensure it's not chosen as a neighbor
        else:
            penalty = 0
        return np.linalg.norm(x1[:-1] - x2[:-1]) + penalty

    # Initialize the KNN model with the custom distance metric
    regressor = KNeighborsRegressor(n_neighbors=2, metric=custom_distance_metric)

    if GridSearch:
        param_grid = {'n_neighbors': [2, 3, 5, 7]}
        grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)
        grid_search.fit(X_train_transformed, y_train)
        print("Best parameters found by GridSearch:", grid_search.best_params_)
        regressor = grid_search.best_estimator_
    else:
        regressor.fit(X_train_transformed, y_train)

    # Predict on test data
    y_pred = regressor.predict(X_test_transformed)
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r_squared}")
    print(f"Mean Absolute Error: {mae}")

    # Plot Predicted vs. Actual Values
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs. Actual Values")
    plt.show()

    # Plot Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, residuals, alpha=0.3)
    plt.hlines(0, y_test.min(), y_test.max(), colors="r", linestyles="dashed")
    plt.xlabel("Actual")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")
    plt.show()

    # Impute the missing data
    imputed_X = imputed_df.drop(columns=["new_oms"])
    imputed_X_transformed = preprocessor.transform(imputed_X)

    if apply_pca:
        imputed_X_transformed = pca.transform(imputed_X_transformed)

    imputed_df["predicted_oms"] = regressor.predict(imputed_X_transformed)
    
    return imputed_df


def knn_model_with_pca_mae_per_year(training_df, scaler, df_estimeres, year, GridSearch=False, apply_pca=True, n_components=6):
    """
    Trains a K-Nearest Neighbors model for predicting new_oms values with optional PCA for dimensionality reduction
    and optional GridSearch for hyperparameter tuning. Calculates MAE per year after final model training.
    
    Parameters:
    training_df (pd.DataFrame): DataFrame containing the training data.
    scaler (object): Scaler object for numerical features (e.g., StandardScaler, RobustScaler).
    df_estimeres (pd.DataFrame): DataFrame containing the data to be imputed.
    GridSearch (bool): Whether to perform GridSearch for hyperparameter tuning. Default is True.
    apply_pca (bool): Whether to apply PCA for dimensionality reduction. Default is True.
    n_components (int, float, or None): Number of components to keep after applying PCA. If None, it will not reduce dimensions.
    
    Returns:
    pd.DataFrame: DataFrame with predicted new_oms values.
    """
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.neighbors import KNeighborsRegressor
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Make copies of the input DataFrames
    df = training_df.copy()
    imputed_df = df_estimeres.copy()
    
    # Define categorical and numerical columns
    categorical_columns = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    df[categorical_columns] = df[categorical_columns].astype(str)
    imputed_df[categorical_columns] = imputed_df[categorical_columns].astype(str)

    columns_to_fill = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    numeric_columns_to_fill = [
        "inntekt_delta_oms", "emp_delta_oms", "befolkning_delta_oms", 
        "inflation_rate_oms", "gjeldende_bdr_syss", "new_oms_trendForecast",
        'oms_syssmean_basedOn_naring', 'oms_syssmean_basedOn_naring_kommune'
    ]

    # Fill missing values
    df[columns_to_fill] = df[columns_to_fill].fillna('missing')
    imputed_df[columns_to_fill] = imputed_df[columns_to_fill].fillna('missing')
    df[numeric_columns_to_fill] = df[numeric_columns_to_fill].fillna(0)
    imputed_df[numeric_columns_to_fill] = imputed_df[numeric_columns_to_fill].fillna(0)

    # Preprocessing pipeline
    categorical_features = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    numerical_features = [
        "inntekt_delta_oms", "emp_delta_oms", "befolkning_delta_oms", 
        "inflation_rate_oms", "gjeldende_bdr_syss", "new_oms_trendForecast", 
        'oms_syssmean_basedOn_naring', 'oms_syssmean_basedOn_naring_kommune'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, numerical_features),
            ("cat", OneHotEncoder(categories="auto", handle_unknown="ignore"), categorical_features),
        ]
    )
    
    # Calculate MAE per year
    mae_per_year = []
    start_year = df['year'].min()
    end_year = df['year'].max()
    
    for current_year in range(start_year, end_year + 1):
        # Split into training and testing based on the year
        train_df = df[df["year"] < current_year]
        test_df = df[df["year"] == current_year]
        
        if train_df.empty or test_df.empty:
            print(f"No data available for training or testing for the year {current_year}. Skipping this year.")
            continue

        # Define features and target for train and test
        X_train = train_df.drop(columns=["new_oms", "year"])
        y_train = train_df["new_oms"]
        X_test = test_df.drop(columns=["new_oms", "year"])
        y_test = test_df["new_oms"]

        # Transform the data
        preprocessor.fit(X_train)
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        if apply_pca:
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train_transformed)
            X_test_pca = pca.transform(X_test_transformed)
            X_train_transformed = X_train_pca
            X_test_transformed = X_test_pca

        # Define and fit the model
        if GridSearch:
            param_grid = {'n_neighbors': [2, 3, 5, 7]}
            grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, scoring='neg_mean_squared_error', cv=5)
            grid_search.fit(X_train_transformed, y_train)
            model = grid_search.best_estimator_
        else:
            model = KNeighborsRegressor(n_neighbors=2)
            model.fit(X_train_transformed, y_train)

        # Make predictions and calculate MAE
        y_pred = model.predict(X_test_transformed)
        mae = mean_absolute_error(y_test, y_pred)
        mae_per_year.append((current_year, mae))
        print(f"Year {current_year}: MAE = {mae}")

    # Print MAE for each year
    print("\nMAE per Year:")
    for year, mae in mae_per_year:
        print(f"{year}: {mae}")

    # Optionally, plot MAE over the years
    if mae_per_year:
        years, mae_values = zip(*mae_per_year)
        plt.figure(figsize=(10, 5))
        plt.plot(years, mae_values, marker='o')
        plt.xlabel("Year")
        plt.ylabel("Mean Absolute Error (MAE)")
        plt.title("MAE per Year")
        plt.grid(True)
        plt.show()

    # Final training with the most recent data
    # Impute the missing data for the current year
    imputed_X = imputed_df.drop(columns=["new_oms"])
    imputed_X_transformed = preprocessor.transform(imputed_X)

    if apply_pca:
        imputed_X_transformed = pca.transform(imputed_X_transformed)

    imputed_df["predicted_oms"] = model.predict(imputed_X_transformed)
    
    return imputed_df


def evaluate_year_based_mae(training_df, scaler, df_estimeres, start_year=2017, end_year=2023, GridSearch=True):
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    import matplotlib.pyplot as plt

    # Prepare categorical and numerical features
    categorical_features = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    numerical_features = [
        "inntekt_delta_oms", "emp_delta_oms", "befolkning_delta_oms", 
        "inflation_rate_oms", "gjeldende_bdr_syss", "new_oms_trendForecast",
        'oms_syssmean_basedOn_naring', 'oms_syssmean_basedOn_naring_kommune'
    ]
    
    # Preprocessor setup
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, numerical_features),
            ("cat", OneHotEncoder(categories="auto", handle_unknown="ignore"), categorical_features),
        ]
    )

    mae_per_year = []
    for year in range(start_year, end_year + 1):
        # Split into training and testing based on the year
        train_df = training_df[training_df["year"] < year]
        test_df = training_df[training_df["year"] == year]
        
        if train_df.empty or test_df.empty:
            print(f"No data available for training or testing for the year {year}. Skipping this year.")
            continue

        X_train = train_df.drop(columns=["new_oms", "year"])
        y_train = train_df["new_oms"]
        X_test = test_df.drop(columns=["new_oms", "year"])
        y_test = test_df["new_oms"]

        # Transform the data
        preprocessor.fit(X_train)
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Define and fit the model
        if GridSearch:
            from sklearn.model_selection import GridSearchCV
            param_grid = {'n_neighbors': [2, 3, 5, 7]}
            grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, scoring='neg_mean_squared_error', cv=5)
            grid_search.fit(X_train_transformed, y_train)
            model = grid_search.best_estimator_
        else:
            model = KNeighborsRegressor(n_neighbors=2)
            model.fit(X_train_transformed, y_train)

        # Make predictions and calculate MAE
        y_pred = model.predict(X_test_transformed)
        mae = mean_absolute_error(y_test, y_pred)
        mae_per_year.append((year, mae))
        print(f"Year {year}: MAE = {mae}")

    # Print MAE for each year
    print("MAE per Year:")
    for year, mae in mae_per_year:
        print(f"{year}: {mae}")

    # Optionally, plot MAE over the years
    years, mae_values = zip(*mae_per_year)
    plt.plot(years, mae_values, marker='o')
    plt.xlabel("Year")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.title("MAE per Year")
    plt.show()






def nn_model_1(training_df, scaler, epochs_number, batch_size, df_estimeres, GridSearch=True):
    """
    Trains a neural network model for predicting new_oms values with an optional GridSearch for hyperparameter tuning.

    Parameters:
    training_df (pd.DataFrame): DataFrame containing the training data.
    scaler (object): Scaler object for numerical features (e.g., StandardScaler, RobustScaler).
    epochs_number (int): Number of epochs for training the neural network.
    batch_size (int): Batch size for training the neural network.
    df_estimeres (pd.DataFrame): DataFrame containing the data to be imputed.
    GridSearch (bool): Whether to perform GridSearch for hyperparameter tuning. Default is True.

    Returns:
    pd.DataFrame: DataFrame with predicted new_oms values.
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from scikeras.wrappers import KerasRegressor
    import tensorflow as tf
    import matplotlib.pyplot as plt

    def build_nn_model(input_shape, learning_rate=0.001, dropout_rate=0.5, neurons_layer1=64, neurons_layer2=32, activation='relu', optimizer='adam'):
        """
        Builds and compiles a neural network model.

        Parameters:
        input_shape (int): Number of input features.
        learning_rate (float): Learning rate for the optimizer.
        dropout_rate (float): Dropout rate for regularization.
        neurons_layer1 (int): Number of neurons in the first layer.
        neurons_layer2 (int): Number of neurons in the second layer.
        activation (str): Activation function to use.
        optimizer (str): Optimizer to use.

        Returns:
        tf.keras.Model: Compiled neural network model.
        """
        
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(neurons_layer1, input_shape=(input_shape,), activation=activation, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.Dense(neurons_layer2, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mse'])
        return model

    # Prepare the data
    df = training_df.copy()
    imputed_df = df_estimeres.copy()

    # Fill NaN values in specified columns
    columns_to_fill = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    numeric_columns_to_fill = [
        "inntekt_delta_oms",
        "emp_delta_oms",
        "befolkning_delta_oms",
        "inflation_rate_oms",
        "gjeldende_bdr_syss",
        "new_oms_trendForecast",
        'oms_syssmean_basedOn_naring',
        'oms_syssmean_basedOn_naring_kommune'
    ]

    df[columns_to_fill] = df[columns_to_fill].fillna('missing')
    imputed_df[columns_to_fill] = imputed_df[columns_to_fill].fillna('missing')
    df[numeric_columns_to_fill] = df[numeric_columns_to_fill].fillna(0)
    imputed_df[numeric_columns_to_fill] = imputed_df[numeric_columns_to_fill].fillna(0)

    # Convert categorical columns to category type
    categorical_columns = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    for col in categorical_columns:
        df[col] = df[col].astype("category")

    # Define features and target
    X = df.drop(columns=["new_oms"])
    y = df["new_oms"]

    # Define preprocessor
    categorical_features = categorical_columns
    numerical_features = [
        "inntekt_delta_oms",
        "emp_delta_oms",
        "befolkning_delta_oms",
        "inflation_rate_oms",
        "gjeldende_bdr_syss",
        "new_oms_trendForecast",
        'oms_syssmean_basedOn_naring',
        'oms_syssmean_basedOn_naring_kommune'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, numerical_features),
            ("cat", OneHotEncoder(categories="auto", handle_unknown="ignore"), categorical_features),
        ]
    )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Transform the data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    input_shape = X_train_transformed.shape[1]

    # Wrap the model with KerasRegressor
    nn_model = KerasRegressor(build_fn=build_nn_model, input_shape=input_shape, epochs=epochs_number, batch_size=batch_size, verbose=1)

    # Define early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    if GridSearch:
        # Perform Grid Search for hyperparameter tuning
        param_grid = {
            'epochs': [100, 200, 300, 400],
            'batch_size': [10, 32, 64, 128],
            'learning_rate': [0.001, 0.01, 0.1],
            'dropout_rate': [0.3, 0.5, 0.7],
            'neurons_layer1': [32, 64, 128],
            'neurons_layer2': [16, 32, 64],
            'activation': ['relu', 'tanh'],
            'optimizer': ['adam', 'sgd', 'rmsprop']
        }
        grid_search = GridSearchCV(estimator=nn_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)
        grid_search.fit(X_train_transformed, y_train, callbacks=[early_stopping])
        print("Best parameters found by GridSearch:", grid_search.best_params_)
        nn_model = grid_search.best_estimator_
    else:
        # Train the model with provided parameters
        history = nn_model.fit(X_train_transformed, y_train, epochs=epochs_number, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])

    # Predict on test data
    y_pred = nn_model.predict(X_test_transformed).flatten()  # Ensure y_pred is 1-dimensional

    # Set negative predictions to zero
    y_pred = np.maximum(y_pred, 0)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Median Absolute Error (MedAE): {medae}")
    print(f"R-squared score: {r_squared}")
    
    X_test['n3'] = X_test['nacef_5'].str[:4]
    
    # Evaluate performance based on the n3 class
    results = pd.DataFrame({'n3': X_test['n3'], 'actual': y_test, 'predicted': y_pred})
    
    # Define the n3 categories to exclude
    n3_to_exclude = ['45.1', '45.2', '46.3', '46.4', '46.5', '46.7', '46.9', '10.4', '02.4']

    # Check if there are any n3 categories not in the excluded list
    if not results['n3'].isin(n3_to_exclude).all():
        # Filter out the rows where the n3 is in the excluded list
        filtered_results = results[~results['n3'].isin(n3_to_exclude)]

        # Extract the actual and predicted values after filtering
        filtered_y_test = filtered_results['actual']
        filtered_y_pred = filtered_results['predicted']

        # Recalculate the evaluation metrics excluding the specified n3 categories
        filtered_mse = mean_squared_error(filtered_y_test, filtered_y_pred)
        filtered_mae = mean_absolute_error(filtered_y_test, filtered_y_pred)
        filtered_medae = median_absolute_error(filtered_y_test, filtered_y_pred)
        filtered_r_squared = r2_score(filtered_y_test, filtered_y_pred)
        filtered_rmse = np.sqrt(filtered_mse)

        # Print out the filtered metrics
        print(f"Filtered Mean Squared Error (MSE): {filtered_mse}")
        print(f"Filtered Mean Absolute Error (MAE): {filtered_mae}")
        print(f"Filtered Median Absolute Error (MedAE): {filtered_medae}")
        print(f"Filtered R-squared score: {filtered_r_squared}")
        print(f"Filtered Root Mean Squared Error (RMSE): {filtered_rmse}")
    else:
        print("No valid n3 categories found after exclusion. Skipping filtered metrics calculation.")
    
    metrics_per_n3 = results.groupby('n3').apply(lambda group: pd.Series({
        'mse': mean_squared_error(group['actual'], group['predicted']),
        'r_squared': r2_score(group['actual'], group['predicted']),
        'mae': mean_absolute_error(group['actual'], group['predicted'])
    })).reset_index()
    
    print("Metrics per 'n3':")
    print(metrics_per_n3)
    
        # Plot loss curve
    if not GridSearch:
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.show()

    # Plot Predicted vs. Actual Values
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs. Actual Values")
    plt.show()

    # Plot Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, residuals, alpha=0.3)
    plt.hlines(0, y_test.min(), y_test.max(), colors="r", linestyles="dashed")
    plt.xlabel("Actual")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")
    plt.show()

    # Impute the missing data
    imputed_X = imputed_df.drop(columns=["new_oms"])
    imputed_X_transformed = preprocessor.transform(imputed_X)
    imputed_df["predicted_oms"] = nn_model.predict(imputed_X_transformed)
    
    return imputed_df


# def nn_model_2(training_df, scaler, epochs_number, batch_size, df_estimeres):
    
#     import pandas as pd
#     import numpy as np
#     import xgboost as xgb
#     from sklearn.model_selection import train_test_split, learning_curve
#     from sklearn.metrics import accuracy_score, classification_report
#     from sklearn.preprocessing import OneHotEncoder
#     from sklearn.compose import ColumnTransformer
#     from sklearn.pipeline import Pipeline
#     from sklearn.impute import SimpleImputer
#     import matplotlib.pyplot as plt
    

#     def build_nn_model(input_shape):
#         model = tf.keras.models.Sequential()
#         model.add(tf.keras.layers.Dense(64, input_shape=(input_shape,), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
#         model.add(tf.keras.layers.Dropout(0.5))
#         model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
#         model.add(tf.keras.layers.Dropout(0.5))
#         model.add(tf.keras.layers.Dense(1, activation='linear'))
#         model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
#         return model

#     df = training_df.copy()
#     imputed_df = df_estimeres.copy()
 
#     columns_to_fill = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
#     numeric_columns_to_fill = ["inntekt_delta_oms",
#         "emp_delta_oms",
#         "befolkning_delta_oms",
#         "inflation_rate_oms",
#         "gjeldende_bdr_syss",
#         "new_oms_trendForecast",
#         'oms_syssmean_basedOn_naring',
#         'oms_syssmean_basedOn_naring_kommune']

#     # Fill NaN values with 'missing' for the specified columns
#     df[columns_to_fill] = df[columns_to_fill].fillna('missing')
#     imputed_df[columns_to_fill] = imputed_df[columns_to_fill].fillna('missing')
    
#     df[numeric_columns_to_fill] = df[numeric_columns_to_fill].fillna(0)
#     imputed_df[numeric_columns_to_fill] = imputed_df[numeric_columns_to_fill].fillna(0)

#     categorical_columns = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
#     for col in categorical_columns:
#         df[col] = df[col].astype("category")

#     categorical_features = categorical_columns

#     numerical_features = [
#         "inntekt_delta_oms",
#         "emp_delta_oms",
#         "befolkning_delta_oms",
#         "inflation_rate_oms",
#         "gjeldende_bdr_syss",
#         "new_oms_trendForecast",
#         'oms_syssmean_basedOn_naring',
#         'oms_syssmean_basedOn_naring_kommune'
#     ]

#     X = df.drop(columns=["new_oms"])
#     y = df["new_oms"]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", scaler, numerical_features),
#             ("cat", OneHotEncoder(categories="auto", handle_unknown="ignore"), categorical_features),
#         ]
#     )

#     X_train_transformed = preprocessor.fit_transform(X_train)
#     X_test_transformed = preprocessor.transform(X_test)
#     input_shape = X_train_transformed.shape[1]

#     model = build_nn_model(input_shape)
    
#     # Define learning rate scheduler callback
#     def scheduler(epoch, lr):
#         if epoch < 60:
#             return lr
#         else:
#             return float(lr * tf.math.exp(-0.1))
    
#     lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
#     early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#     history = model.fit(
#         X_train_transformed, y_train,
#         validation_split=0.2,
#         epochs=epochs_number,
#         batch_size=batch_size,
#         callbacks=[early_stopping, lr_scheduler],
#         verbose=1
#     )

#     history_dict = history.history
#     plt.plot(history_dict['loss'], label='train loss')
#     plt.plot(history_dict['val_loss'], label='validation loss')
#     plt.legend()
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Training vs. Validation Loss')
#     plt.show()

#     y_pred = model.predict(X_test_transformed).flatten()  # Ensure y_pred is 1-dimensional

#     # Set negative predictions to zero
#     y_pred = np.maximum(y_pred, 0)

#     mse = mean_squared_error(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     medae = median_absolute_error(y_test, y_pred)
#     r_squared = r2_score(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     mean_y_test = np.mean(y_test)

#     print(f"Mean Squared Error (MSE): {mse}")
#     print(f"Mean Absolute Error (MAE): {mae}")
#     print(f"Median Absolute Error (MedAE): {medae}")
#     print(f"R-squared score: {r_squared}")

#     plt.figure(figsize=(10, 5))
#     plt.scatter(y_test, y_pred, alpha=0.3)
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
#     plt.xlabel("Actual")
#     plt.ylabel("Predicted")
#     plt.title("Predicted vs. Actual Values")
#     plt.show()

#     residuals = y_test - y_pred
#     plt.figure(figsize=(10, 5))
#     plt.scatter(y_test, residuals, alpha=0.3)
#     plt.hlines(0, y_test.min(), y_test.max(), colors="r", linestyles="dashed")
#     plt.xlabel("Actual")
#     plt.ylabel("Residuals")
#     plt.title("Residuals Plot")
#     plt.show()

#     # imputed_X = imputed_df.drop(columns=["new_oms"])
#     # imputed_X_transformed = preprocessor.transform(imputed_X)
#     # imputed_df["predicted_oms"] = model_pipeline.named_steps['nn_model'].predict(imputed_X_transformed)
    
#     return imputed_df

def nn_model(training_df, scaler, epochs_number, batch_size, df_estimeres, GridSearch=False):
    
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    import matplotlib.pyplot as plt
    from IPython.display import clear_output

    def build_nn_model(input_shape):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(64, input_shape=(input_shape,), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
        return model

    df = training_df.copy()
    imputed_df = df_estimeres.copy()
 
    columns_to_fill = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    numeric_columns_to_fill = ["inntekt_delta_oms",
        "emp_delta_oms",
        "befolkning_delta_oms",
        "inflation_rate_oms",
        "gjeldende_bdr_syss",
        "new_oms_trendForecast",
        'oms_syssmean_basedOn_naring',
        'oms_syssmean_basedOn_naring_kommune']

    # Fill NaN values with 'missing' for the specified columns
    df[columns_to_fill] = df[columns_to_fill].fillna('missing')
    imputed_df[columns_to_fill] = imputed_df[columns_to_fill].fillna('missing')
    
    df[numeric_columns_to_fill] = df[numeric_columns_to_fill].fillna(0)
    imputed_df[numeric_columns_to_fill] = imputed_df[numeric_columns_to_fill].fillna(0)

    categorical_columns = ["nacef_5", "tmp_sn2007_5", "b_kommunenr"]
    for col in categorical_columns:
        df[col] = df[col].astype("category")

    categorical_features = categorical_columns

    numerical_features = [
        "inntekt_delta_oms",
        "emp_delta_oms",
        "befolkning_delta_oms",
        "inflation_rate_oms",
        "gjeldende_bdr_syss",
        "new_oms_trendForecast",
        'oms_syssmean_basedOn_naring',
        'oms_syssmean_basedOn_naring_kommune'
    ]

    X = df.drop(columns=["new_oms"])
    y = df["new_oms"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, numerical_features),
            ("cat", OneHotEncoder(categories="auto", handle_unknown="ignore"), categorical_features),
        ]
    )

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    input_shape = X_train_transformed.shape[1]

    model = build_nn_model(input_shape)
    
    # Define learning rate scheduler callback
    def scheduler(epoch, lr):
        if epoch < 60:
            return lr
        else:
            return float(lr * tf.math.exp(-0.1))
    
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Real-time plotting function
    def live_plotting(epoch, logs):
        loss.append(logs['loss'])
        val_loss.append(logs['val_loss'])
        epochs.append(epoch)
        
        clear_output(wait=True)  # Clears the output of the cell
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

    # Lists to hold the values for plotting
    loss, val_loss, epochs = [], [], []
    plot_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=live_plotting)
    
    history = model.fit(
        X_train_transformed, y_train,
        validation_split=0.2,
        epochs=epochs_number,
        batch_size=batch_size,
        callbacks=[early_stopping, lr_scheduler, plot_callback],  # Add the plotting callback
        verbose=1
    )

    y_pred = model.predict(X_test_transformed).flatten()  # Ensure y_pred is 1-dimensional

    # Set negative predictions to zero
    y_pred = np.maximum(y_pred, 0)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    mean_y_test = np.mean(y_test)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Median Absolute Error (MedAE): {medae}")
    print(f"R-squared score: {r_squared}")
    
    X_test['n3'] = X_test['nacef_5'].str[:4]
    
    # Evaluate performance based on the n3 class
    results = pd.DataFrame({'n3': X_test['n3'], 'actual': y_test, 'predicted': y_pred})
    
    # Define the n3 categories to exclude
    n3_to_exclude = ['45.1', '45.2', '46.3', '46.4', '46.5', '46.7', '46.9', '10.4', '02.4']

    # Check if there are any n3 categories not in the excluded list
    if not results['n3'].isin(n3_to_exclude).all():
        # Filter out the rows where the n3 is in the excluded list
        filtered_results = results[~results['n3'].isin(n3_to_exclude)]

        # Extract the actual and predicted values after filtering
        filtered_y_test = filtered_results['actual']
        filtered_y_pred = filtered_results['predicted']

        # Recalculate the evaluation metrics excluding the specified n3 categories
        filtered_mse = mean_squared_error(filtered_y_test, filtered_y_pred)
        filtered_mae = mean_absolute_error(filtered_y_test, filtered_y_pred)
        filtered_medae = median_absolute_error(filtered_y_test, filtered_y_pred)
        filtered_r_squared = r2_score(filtered_y_test, filtered_y_pred)
        filtered_rmse = np.sqrt(filtered_mse)

        # Print out the filtered metrics
        print(f"Filtered Mean Squared Error (MSE): {filtered_mse}")
        print(f"Filtered Mean Absolute Error (MAE): {filtered_mae}")
        print(f"Filtered Median Absolute Error (MedAE): {filtered_medae}")
        print(f"Filtered R-squared score: {filtered_r_squared}")
        print(f"Filtered Root Mean Squared Error (RMSE): {filtered_rmse}")
    else:
        print("No valid n3 categories found after exclusion. Skipping filtered metrics calculation.")
    
    metrics_per_n3 = results.groupby('n3').apply(lambda group: pd.Series({
        'mse': mean_squared_error(group['actual'], group['predicted']),
        'r_squared': r2_score(group['actual'], group['predicted']),
        'mae': mean_absolute_error(group['actual'], group['predicted'])
    })).reset_index()
    
    print("Metrics per 'n3':")
    print(metrics_per_n3)

    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs. Actual Values")
    plt.show()

    residuals = y_test - y_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, residuals, alpha=0.3)
    plt.hlines(0, y_test.min(), y_test.max(), colors="r", linestyles="dashed")
    plt.xlabel("Actual")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")
    plt.show()

    # Predict values for the imputed_df
    imputed_X = imputed_df.drop(columns=["new_oms"])
    imputed_X_transformed = preprocessor.transform(imputed_X)
    imputed_df["predicted_oms"] = model.predict(imputed_X_transformed)

    return imputed_df




def xgboost_n3_klass(df):
    """
    Trains an XGBoost classifier to predict 'n3' categories with preprocessing for numerical and categorical data.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data with 'n3' as the target variable.

    Returns:
    None
    """
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, learning_curve
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    import matplotlib.pyplot as plt

    # Filter out sparse classes
    min_samples_per_class = 50
    value_counts = df['n3'].value_counts()
    to_remove = value_counts[value_counts < min_samples_per_class].index
    df = df[~df['n3'].isin(to_remove)]

    # Convert target to integer labels and store the mapping
    labels, unique = pd.factorize(df['n3'])
    df['n3_encoded'] = labels
    n3_mapping = dict(zip(labels, unique))

    # Identify categorical and numerical columns excluding the target
    non_feature_cols = ['n3']  # 'n3' is now not a feature
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in non_feature_cols]
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    numerical_cols.remove('n3_encoded')  # Assume 'n3_encoded' is the new target variable

    # Preprocessing for numerical data: simple imputer with median strategy
    numerical_transformer = SimpleImputer(strategy='median')

    # Preprocessing for categorical data: impute missing values and apply one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Define the XGBoost classifier
    model = xgb.XGBClassifier(objective='multi:softprob', random_state=42, eval_metric='mlogloss')

    # Create a pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Prepare data
    X = df.drop(non_feature_cols + ['n3_encoded'], axis=1)
    y = df['n3_encoded']  # Correct target variable

    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Fit the model
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)

    # Generate a mapping from labels to original 'n3' categories if not done previously
    n3_mapping = {idx: label for idx, label in enumerate(pd.unique(df['n3']))}

    # Safely convert predictions and true values back to original labels using the mapping
    y_pred_labels = [n3_mapping.get(label, 'Unknown') for label in y_pred]
    y_test_labels = [n3_mapping.get(label, 'Unknown') for label in y_test]

    # Print accuracy and classification report using the original 'n3' labels
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test_labels, y_pred_labels))
    
    # Plot learning curve
    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        """
        Generate a simple plot of the test and training learning curve.

        Parameters:
        estimator: object type that implements the "fit" and "predict" methods
        title: string, Title for the chart
        X: array-like, shape (n_samples, n_features), Training vector
        y: array-like, shape (n_samples) or (n_samples, n_features), Target relative to X for classification or regression
        ylim: tuple, shape (ymin, ymax), optional, Defines minimum and maximum y-values plotted
        cv: int, cross-validation generator or an iterable, optional, Determines the cross-validation splitting strategy
        n_jobs: int or None, optional, Number of jobs to run in parallel
        train_sizes: array-like, shape (n_ticks,), Relative or absolute numbers of training examples that will be used to generate the learning curve

        Returns:
        plt: Matplotlib plot object
        """
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.legend(loc="best")
        return plt

    plot_learning_curve(clf, "Learning Curve for XGBoost Classifier", X_train, y_train, cv=5)
    plt.show()




def knn_n3_klass(df):
    """
    Trains a K-Nearest Neighbors (KNN) classifier to predict 'n3' categories with preprocessing 
    for numerical and categorical data.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data with 'n3' as the target variable.

    Returns:
    None
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, learning_curve
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    import matplotlib.pyplot as plt

    # Filter out sparse classes
    min_samples_per_class = 100
    value_counts = df['n3'].value_counts()
    to_remove = value_counts[value_counts < min_samples_per_class].index
    df = df[~df['n3'].isin(to_remove)]

    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Specify columns to exclude from features
    non_feature_cols = ['n3']
    categorical_cols = [col for col in categorical_cols if col not in non_feature_cols]

    # Preprocessing for numerical data: impute missing values and scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())  # Scale features
    ])

    # Preprocessing for categorical data: impute missing values and apply one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Define the KNN classifier
    model = KNeighborsClassifier(n_neighbors=5)

    # Create a pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Prepare data
    X = df.drop(non_feature_cols, axis=1)
    y = df['n3']

    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Fit the model
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    # Plot learning curve
    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        """
        Generate a simple plot of the test and training learning curve.

        Parameters:
        estimator: object type that implements the "fit" and "predict" methods
        title: string, Title for the chart
        X: array-like, shape (n_samples, n_features), Training vector
        y: array-like, shape (n_samples) or (n_samples, n_features), Target relative to X for classification or regression
        ylim: tuple, shape (ymin, ymax), optional, Defines minimum and maximum y-values plotted
        cv: int, cross-validation generator or an iterable, optional, Determines the cross-validation splitting strategy
        n_jobs: int or None, optional, Number of jobs to run in parallel
        train_sizes: array-like, shape (n_ticks,), Relative or absolute numbers of training examples that will be used to generate the learning curve

        Returns:
        plt: Matplotlib plot object
        """
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.legend(loc="best")
        return plt

    plot_learning_curve(clf, "Learning Curve for KNN Classifier", X_train, y_train, cv=5)
    plt.show()

    
    
def test_results(df, aar):
    # Convert aar to an integer if it's not already
    year = int(aar)
    
    # Determine the correct file name based on the value of aar
    # if year < 2022:
    #     fil_navn = 'statistikkfil_bedrifter_nr.parquet'
    # else:
    #     fil_navn = 'statistiske_foretak_bedrifter.parquet'
    
    # Define the file path based on the determined file name
    fil_path = [
        f
        for f in fs.glob(
            f"gs://ssb-strukt-naering-data-produkt-prod/naringer/klargjorte-data/statistikkfiler/aar={aar}/statistikkfil_bedrifter_nr.parquet"
        )
        if f.endswith(".parquet")
    ]

    # Use the ParquetDataset to read multiple files
    dataset = pq.ParquetDataset(fil_path, filesystem=fs)
    table = dataset.read()

    # Convert to Pandas DataFrame
    bedrift_2 = table.to_pandas()
    
    bedrift_2.columns = bedrift_2.columns.str.lower()  # Convert column names to lower case

    # change pd option to show all columns
    pd.set_option("display.max_columns", None)

    bedrift_2 = bedrift_2[['orgnr_bedrift', 'omsetning', 'nopost_driftskostnader']]
    
    bedrift_1 = df[['v_orgnr', 'oms', 'new_drkost', 'regtype']]

    # rename 
    bedrift_1.rename(columns={"v_orgnr": "orgnr_bedrift"}, inplace=True)
    print(bedrift_1.shape)
    test = bedrift_1.merge(bedrift_2, on='orgnr_bedrift', how='left')
    test = test.drop_duplicates()
    test = test.dropna()
    
    # Calculate the absolute difference
    test['oms_diff'] = (test['oms'] - test['omsetning']).abs()

    # Sort the DataFrame by the 'oms_diff' column in descending order
    test_sorted = test.sort_values(by='oms_diff', ascending=False)

    # Display the sorted DataFrame
    test_sorted.head()

    # create new df where regtype == 02

    test_02 = test_sorted[test_sorted['regtype'] == '02']
    
    # Assuming your DataFrame is named 'test' and has columns 'oms' and 'omsetning'
    oms = test['oms']
    omsetning = test['omsetning']

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(omsetning, oms)
    print(f'Mean Absolute Error for entire delreg: {mae}')

    # Calculate R-squared (R²)
    r2 = r2_score(omsetning, oms)
    print(f'R² Score for entire delreg: {r2}')
    
    
    print(f'-----------------------------------')
    
    # Assuming your DataFrame is named 'test' and has columns 'oms' and 'omsetning'
    oms = test_02['oms']
    omsetning = test_02['omsetning']

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(omsetning, oms)
    print(f'Mean Absolute Error for reg_type 02: {mae}')

    # Calculate R-squared (R²)
    r2 = r2_score(omsetning, oms)
    print(f'R² Score for reg_type 02: {r2}')


def fetch_foretak_data(aar):
    
        # Convert aar to an integer if it's not already
    year = int(aar)
    
    # Determine the correct file name based on the value of aar
    # if year < 2022:
    #     fil_navn = 'statistikkfil_foretak_pub.parquet'
    # else:
    #     fil_navn = 'statistiske_foretak_foretak.parquet'
    
    # Define the file path based on the determined file name
    fil_path = [
        f
        for f in fs.glob(
            f"gs:// ssb-strukt-naering-data-produkt-prod/naringer/klargjorte-data/statistikkfiler/aar={aar}/statistikkfil_foretak_pub.parquet"
        )
        if f.endswith(".parquet")
    ]

    # Use the ParquetDataset to read multiple Parquet files into a single Arrow Table
    dataset = pq.ParquetDataset(fil_path, filesystem=fs)
    table = dataset.read()

    # Convert the Arrow Table into a Pandas DataFrame
    foretak_pub = table.to_pandas()

    # Create a new column 'n3' extracting the first four characters from 'naring_f' column
    # Create a new column 'n2' extracting the first two characters from 'naring_f' column
    foretak_pub['n3'] = foretak_pub['naring_f'].str[:4]
    foretak_pub['n2'] = foretak_pub['naring_f'].str[:2]

    # Filter data where 'n2' indicates specific industry codes relevant to the analysis
    foretak_varendel = foretak_pub[(foretak_pub['n2'] == '45') | (foretak_pub['n2'] == '46') | (foretak_pub['n2'] == '47')]

    # Select only the relevant columns for further processing
    foretak_varendel = foretak_varendel[['orgnr_foretak', 'naring_f', 'n2', 'n3', 'bearbeidingsverdi',
                                         'produktinnsats', 'produksjonsverdi', 'omsetning', 
                                         'nopost_driftsresultat', 'nopost_driftskostnader',
                                         'nopost_driftsinntekter', 'sysselsetting_syss']]
    
    return foretak_pub, foretak_varendel


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

def lstm_model(training_df):
    """
    Trains an LSTM model for predicting new_oms values with hierarchical modeling based on v_orgnr, nace, and kommunenr.

    Parameters:
    training_df (pd.DataFrame): DataFrame containing the training data.
    scaler (object): Scaler object for numerical features (e.g., StandardScaler, RobustScaler).
    df_estimeres (pd.DataFrame): DataFrame containing the data to be imputed.
    epochs (int): Number of epochs for training the LSTM.
    batch_size (int): Batch size for training the LSTM.

    Returns:
    pd.DataFrame: DataFrame with predicted new_oms values.
    """
    # Prepare data
    df = training_df.copy()
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    # Assume df is your DataFrame and it has been properly sorted and prepared as per the previous steps.
    df = df.sort_values(by=['v_orgnr', 'year'])

    # Create lag features and other engineered features
    df['new_oms_lag1'] = df.groupby('v_orgnr')['new_oms'].shift(1)
    df['new_oms_lag2'] = df.groupby('v_orgnr')['new_oms'].shift(2)

    # Drop rows with NaN values from lag features
    df = df.dropna()

    # Encode categorical variables
    le_nacef_5 = LabelEncoder()
    df['nacef_5_encoded'] = le_nacef_5.fit_transform(df['nacef_5'])

    # Feature selection
    features = ['new_oms_lag1', 'new_oms_lag2', 'inntekt_delta_oms', 'emp_delta_oms', 
                'befolkning_delta_oms', 'inflation_rate_oms', 'gjeldende_bdr_syss', 
                'nacef_5_encoded', 'avg_new_oms_per_gjeldende_bdr_syss', 
                'avg_new_oms_per_gjeldende_bdr_syss_kommunenr']

    X = df[features]
    y = df['new_oms']

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape input for LSTM [samples, time steps, features]
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Split the data into training and validation sets
    train_size = int(len(X_scaled) * 0.8)
    X_train, X_val = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # LSTM Model
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=2, shuffle=False)

    # Predict on validation set
    y_pred = model.predict(X_val)

    # Evaluate model performance
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R^2 Score: {r2}')

    # Plot learning loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.show()



