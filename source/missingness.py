import random
import pandas as pd

def simulate_missingness(df, split_ratio=0.5, block_missingness=True, block_size=10):
    """
    Simulate missingness in a DataFrame by randomly removing values.
    Args:
        df (pd.DataFrame): The DataFrame to simulate missingness on.
        split_ratio (float): The ratio of the DataFrame to use for training.
        block_missingness (bool): If True, apply block missingness.
        block_size (int): The size of the blocks for block missingness.
    Returns:
        pd.DataFrame: The training DataFrame.
        pd.DataFrame: The validation DataFrame.
    """
    if block_missingness:
        list_col_blocks = []
        for col in df['columnId'].unique():
            df_col = df[df['columnId'] == col]
            for i in range(0, len(df_col), block_size):
                block = df_col.iloc[i:i + block_size]
                list_col_blocks.append(block)
        random.shuffle(list_col_blocks)
        df_shuffled = pd.concat(list_col_blocks).reset_index(drop=True)
    else:
        df_shuffled = df.sample(frac=1).reset_index(drop=True)
    split_index = int(split_ratio * len(df_shuffled))
    training_df = df_shuffled[:split_index]
    validation_df = df_shuffled[split_index:]
    return training_df, validation_df
