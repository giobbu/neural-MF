from source.generator import generate_synthetic_data
from source.missingness import create_block_missingness
from source.model import NeuralALS
from source.utils import plot_filled_by_idx

from config.setting import Config
params = Config()

def main():
    # Generate training and validation data
    df = generate_synthetic_data(num_rows=params.NUM_ROWS, num_columns=params.NUM_COLUMNS)
    # Shuffle the DataFrame by blocks and Split the DataFrame into training and validation sets
    training_df, validation_df = create_block_missingness(df, block_size=params.BLOCK_SIZE, split_ratio=params.SPLIT_RATIO)
    # Initialize and build the model
    neural_als = NeuralALS(training_df, validation_df, K=params.LATENT_DIM)
    neural_als.build_model(learning_rate=params.LR,
                            reg=params.REG,
                            loss=params.LOSS,
                            metrics=params.METRICS
                            )
    # Train the model
    _ = neural_als.train(epochs=params.EPOCHS,
                         batch_size=params.BATCH_SIZE
                         )
    # Imputation
    imputed_df = neural_als.fill(validation_df)
    # Plot the training, validation and prediction data
    plot_filled_by_idx(training_df, validation_df, imputed_df, col_idx=params.COL_IDX)

if __name__ == "__main__":
    main()
