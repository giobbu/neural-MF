from pydantic_settings import BaseSettings
from typing import List

class Config(BaseSettings):
    """
    Settings class to manage configuration parameters for the project.
    """
    # Data generation parameters
    NUM_ROWS: int = 10000
    NUM_COLUMNS: int = 5
    # Missingness parameters
    BLOCK_SIZE: int = 10
    SPLIT_RATIO: float = 0.5
    SHOW_MISSING: bool = False
    # Model parameters
    LATENT_DIM: int = 10
    LR: float = 0.001
    REG: float = 0.001
    LOSS: str = 'mse'
    METRICS: List[str] = ['mse']
    # Training parameters
    EPOCHS: int = 10
    BATCH_SIZE: int = 64
    # Plotting parameters
    COL_IDX: int = 3