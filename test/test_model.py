
def test_model(load_data_processed, load_params):
    """
    Test the model training and prediction.
    """
    from source.model import NeuralALS
    import tensorflow as tf
    params = load_params
    training_df, validation_df = load_data_processed
    # Initialize model
    neural_als = NeuralALS(training_df, validation_df, K=params.LATENT_DIM)
    # Build model
    neural_als.build_model(learning_rate=params.LR,
                            reg=params.REG,
                            loss=params.LOSS,
                            metrics=params.METRICS
                            )
    assert neural_als.model is not None, "Model is not built"
    assert isinstance(neural_als.model, tf.keras.Model), "Model is not a Keras model" 