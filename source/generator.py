import pandas as pd
import random

# Generate synthetic timeseries data
def generate_synthetic_data(num_rows=1000, num_columns=10):
    data = []
    for i in range(num_rows):
        for j in range(num_columns):
            data.append([i, j, random.uniform(0, 1)])
    return pd.DataFrame(data, columns=["rowId", "columnId", "OBS"])
    