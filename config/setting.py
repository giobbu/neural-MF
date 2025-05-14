from pydantic_settings import BaseSettings
from typing import List

class Config(BaseSettings):
    """
    Settings class to manage configuration parameters for the project.
    """
    # Data generation parameters
    NUM_ROWS: int = 1000
    NUM_COLUMNS: int = 10
    # Missingness parameters
    BLOCK_SIZE: int = 100
    SPLIT_RATIO: float = 0.8
    # Model parameters
    LATENT_DIM: int = 10
    LR: float = 0.1
    REG: float = 0.001
    LOSS: str = 'mse'
    METRICS: List[str] = ['mse']
    # Training parameters
    EPOCHS: int = 100
    BATCH_SIZE: int = 32
    # Plotting parameters
    COL_IDX: int = 2