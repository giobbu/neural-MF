import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

def show_missing_values(df: pd.DataFrame, training_df: pd.DataFrame, show: bool = True, block_missigness: bool= True, block_size: int = 10, split_ratio: float = 0.5):
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
    plt.title(f'Missing Values Visualization\nBlock Size: {block_size}, Split Ratio: {split_ratio}')
    plt.tight_layout()
    if block_missigness:
        plt.savefig(f'imgs/missing_values_block_{block_size}_split_{split_ratio}.png')
    else:
        # Save the plot with a different name if block_missingness is False
        plt.savefig(f'imgs/missing_values_point_split_{split_ratio}.png')
    if show:
        plt.show()
    else:
        plt.close()
