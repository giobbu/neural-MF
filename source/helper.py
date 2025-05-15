import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

def show_missing_values(df: pd.DataFrame, training_df: pd.DataFrame):
    """
    Visualize the missing values in the DataFrame using missingno library.
    Args:
        df (pd.DataFrame): The original DataFrame with missing values.
        training_df (pd.DataFrame): The DataFrame used for training, containing the missing values.
    """
    training_df_show = training_df.merge(df, on=['rowId', 'columnId'], how='right').drop(columns=['OBS_y'])
    training_df_show.columns = ['rowId', 'columnId', 'OBS']
    training_df_show = training_df_show.pivot(index='rowId', columns='columnId', values='OBS')
    msno.matrix(training_df_show, figsize=(20, 10))
    plt.show()