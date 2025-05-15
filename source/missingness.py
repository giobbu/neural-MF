import random
import pandas as pd

def create_block_missingness(df, block_size=10, split_ratio=0.5):
    """Create block missingness in the DataFrame grouped by 'columnId'."""
    list_col_blocks = []
    for col in df['columnId'].unique():
        df_col = df[df['columnId'] == col]
        for i in range(0, len(df_col), block_size):
            block = df_col.iloc[i:i + block_size]
            list_col_blocks.append(block)
    random.shuffle(list_col_blocks)
    df_shuffled = pd.concat(list_col_blocks).reset_index(drop=True)
    split_index = int(split_ratio * len(df_shuffled))
    training_df = df_shuffled[:split_index]
    validation_df = df_shuffled[split_index:]
    return training_df, validation_df