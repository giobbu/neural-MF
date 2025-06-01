import pandas as pd
import numpy as np
import time

from joblib import Parallel, delayed
from multiprocessing import cpu_count

def generate_user_ids_chunk(chunk_size, num_users, seed=None):
    """
    Generate a chunk of userId strings.
    """
    rng = np.random.default_rng(seed)
    user_ids = ['user_' + str(i) for i in rng.integers(1, num_users, size=chunk_size)]
    return pd.DataFrame({'userId': user_ids})

def generate_user_ids(num_rows, num_users):
    """
    Generate a DataFrame with n userId strings using parallel processing.
    """
    start = time.time()
    num_cores = cpu_count()
    print(f"Using {num_cores} cores for parallel processing.")
    chunk_size = num_rows // num_cores
    remainder = num_rows % num_cores

    # Prepare arguments for each process
    args = [chunk_size] * num_cores
    # Distribute the remainder among the first few chunks
    for i in range(remainder):
        args[i] += 1

    # Assign a unique seed for each process to ensure different random numbers
    seeds = np.random.SeedSequence().spawn(len(args))
    args_with_seeds = [(size, seed) for size, seed in zip(args, seeds)]

    # Generate user IDs in parallel
    dfs = Parallel(n_jobs=num_cores)(delayed(generate_user_ids_chunk)(size, num_users, seed) for size, seed in args_with_seeds)

    df = pd.concat(dfs, ignore_index=True)
    end = time.time()
    print(f"DataFrame with {num_rows} rows and {num_users} unique userIds generated in {end - start:.6f} seconds.")
    return df

def experiment_with_category(df):
    forward_start = time.time()
    df['user_label'] = df['userId'].astype('category').cat.codes
    d = dict(enumerate(df['userId'].astype('category').cat.categories))
    df["userId_back"] = df['user_label'].map(d)
    backward_end = time.time()
    total_time = backward_end - forward_start
    print(f"Total time taken from category dtype method: {backward_end - forward_start:.6f} seconds")
    return total_time

def experiment_with_dict(df):
    forward_start = time.time()
    user2id = {user: idx for idx, user in enumerate(df['userId'].unique())}
    df['user_label'] = df['userId'].map(user2id)
    id2user = {idx: user for user, idx in user2id.items()}
    df["userId_back"] = df['user_label'].map(id2user)
    backward_end = time.time()
    total_time = backward_end - forward_start
    print(f"Total time taken from dictionary method: {backward_end - forward_start:.6f} seconds")
    return total_time

def scalability_experiment():
    """
    Run scalability experiments with different DataFrame sizes and number of unique users.
    """
    num_rows = [100000, 1000000, 10000000]
    num_users = [1000, 10000, 100000]
    for n in num_rows:
        for num_user in num_users:
            print("--"*100)
            print("--"*100)
            print(f"Running experiments with {n} rows")
            print(f"Number of unique users: {num_user}")
            df = generate_user_ids(n, num_user)
            tot_time_cat = experiment_with_category(df)
            tot_time_dict = experiment_with_dict(df)
            # print improvement respect to dictionary method
            improvement = (tot_time_dict - tot_time_cat) / tot_time_dict * 100
            print(f"Improvement of category dtype method over dictionary method: {improvement:.2f}%")

if __name__ == "__main__":
    scalability_experiment()
    print("Scalability experiment completed.")