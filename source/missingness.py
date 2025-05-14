import random
import pandas as pd

def create_block_missingness(df, block_size, split_ratio=0.8):
    " Create block missingness in the DataFrame."
    shuffled_df = []
    for i in range(0, len(df), block_size):
        block = df[i:i + block_size]
        shuffled_df.append(block)
    random.shuffle(shuffled_df)  # Shuffle the blocks
    df = pd.concat(shuffled_df).reset_index(drop=True)
    split_index = int(split_ratio * len(df))
    training_df = df[:split_index]
    validation_df = df[split_index:]
    return training_df, validation_df