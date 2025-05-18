
def test_missingness(load_generator_data, load_params):
    """
    Test the missingness creation function.
    """
    from source.missingness import simulate_missingness
    params = load_params
    df = load_generator_data
    # Create block missingness
    training_df, validation_df = simulate_missingness(df,
        split_ratio=params.SPLIT_RATIO,
        block_missingness=params.BLOCK_MISSINGNESS,
        block_size=params.BLOCK_SIZE
    )
    assert not training_df.empty, "Training DataFrame is empty"
    assert not validation_df.empty, "Validation DataFrame is empty"
    assert len(training_df) + len(validation_df) == len(df), "Total number of rows in training and validation DataFrames does not match the original DataFrame"
    assert training_df.shape[1] == df.shape[1], "Number of columns in training DataFrame does not match the original DataFrame"