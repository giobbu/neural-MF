from source.generator import generate_synthetic_data
from source.missingness import simulate_missingness
from source.model import NeuralMF
from source.utils import plot_filled_by_idx
from source.helper import show_missing_values

from config.setting import Config
params = Config()

def main():

    # Generate training and validation data
    df = generate_synthetic_data(num_rows=params.NUM_ROWS, num_columns=params.NUM_COLUMNS)
    
    # Simulate missingness in the DataFrame
    training_df, validation_df = simulate_missingness(df,
        split_ratio=params.SPLIT_RATIO,
        block_missingness=params.BLOCK_MISSINGNESS,
        block_size=params.BLOCK_SIZE
    )

    # Show the missing values in the DataFrame
    show_missing_values(df, training_df, params.SHOW_MISSING, params.BLOCK_MISSINGNESS, params.BLOCK_SIZE, params.SPLIT_RATIO)

    # Initialize and build the model
    neural_matrix_factor = NeuralMF(training_df, validation_df, K=params.LATENT_DIM)
    neural_matrix_factor.build_model(learning_rate=params.LR,
                            reg=params.REG,
                            loss=params.LOSS,
                            metrics=params.METRICS
                            )
    # Train the model
    _ = neural_matrix_factor.train(epochs=params.EPOCHS,
                         batch_size=params.BATCH_SIZE
                         )
    # Imputation
    imputed_df = neural_matrix_factor.fill(validation_df)
    # Plot the training, validation and prediction data
    plot_filled_by_idx(training_df, validation_df, imputed_df, col_idx=params.COL_IDX)

if __name__ == "__main__":
    main()
