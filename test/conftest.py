import pytest
import numpy as np
import os

@pytest.fixture
def load_generator_data():
    """
    Fixture to load the generator data.
    """
    from source.generator import generate_synthetic_data
    from config.setting import Config
    params = Config()
    df = generate_synthetic_data(num_rows=params.NUM_ROWS, num_columns=params.NUM_COLUMNS)
    return df

@pytest.fixture
def load_params():
    """
    Fixture to load the parameters.
    """
    from config.setting import Config
    params = Config()
    return params

@pytest.fixture
def load_data_processed(load_generator_data):
    """
    Fixture to load the data processor.
    """
    from config.setting import Config
    params = Config()
    from source.missingness import simulate_missingness
    df = load_generator_data
    # Create block missingness
    training_df, validation_df = simulate_missingness(df,
        split_ratio=params.SPLIT_RATIO,
        block_missingness=params.BLOCK_MISSINGNESS,
        block_size=params.BLOCK_SIZE
    )
    return training_df, validation_df