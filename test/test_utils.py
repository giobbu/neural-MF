
def test_filter_col(load_generator_data):
    """
    Test the filter_col function.
    """
    from source.utils import filter_col
    df = load_generator_data
    col = 0
    filtered_df = filter_col(df, col)
    assert filtered_df.shape[0] == df[df['columnId'] == col].shape[0], "Filtered DataFrame does not match expected shape"
    assert filtered_df['columnId'].nunique() == 1, "Filtered DataFrame should have only one unique columnId"

def test_get_segments(load_generator_data):
    """
    Test the get_segments function.
    """
    from source.utils import get_segments, filter_col
    df = load_generator_data
    df_col = filter_col(df, 0)
    segments = get_segments(df_col)
    assert len(segments) > 0, "Segments list is empty"
    for start, end in segments:
        assert start < end, "Start index should be less than end index"
        assert start >= 0 and end <= len(df), "Start and end indices should be within the DataFrame bounds"
