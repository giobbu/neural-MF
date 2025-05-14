import pandas as pd
import matplotlib.pyplot as plt

def filter_col(df: pd.DataFrame, col: int) -> pd.DataFrame:
    " Filter the DataFrame by columnId."
    return df[df['columnId'] == col].sort_values(by='rowId')

def get_segments(df: pd.DataFrame) -> list:
    " Get segments of continuous rowId values in a DataFrame."
    segments = []
    start = 0
    for i in range(1, len(df.rowId)):
        if df.rowId.values[i] != df.rowId.values[i-1] + 1:
            segments.append((start, i))
            start = i
    segments.append((start, len(df.rowId)))
    return segments

def plot_segments(df_col, col_name, segments, color):
    for start, end in segments:
        plt.plot(df_col.rowId.values[start:end], df_col[col_name].values[start:end], color=color)


def plot_filled_by_idx(training_df: pd.DataFrame, validation_df: pd.DataFrame, df_predictions: pd.DataFrame, col_idx: int):
    """
    Plot the training, validation and prediction data.
    """
    # Filter the DataFrames by columnId
    training_col = filter_col(training_df, col_idx)
    validation_col = filter_col(validation_df, col_idx)
    predictions_col = filter_col(df_predictions, col_idx)
    # Check if the DataFrames have the same number of rows
    assert predictions_col.shape[0] == validation_col.shape[0], "Predictions and validation data must have the same number of rows."
    # Get segments of continuous rowId values
    segments_train = get_segments(training_col)
    segments_test = get_segments(validation_col)
    segments_pred = get_segments(predictions_col)
    # Plot training, validation and prediction data
    plt.figure(figsize=(20, 6))
    plot_segments(training_col, 'OBS', segments_train, 'blue')
    plot_segments(validation_col, 'OBS', segments_test, 'orange')
    plot_segments(predictions_col, 'predictions', segments_pred, 'green')
    plt.xlabel('Row ID')
    plt.ylabel('Value')
    plt.title('Training, Validation and Prediction Data')
    plt.legend()
    plt.show()
