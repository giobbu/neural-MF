[![Python Tests](https://github.com/giobbu/neural-MF/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/giobbu/neural-MF/actions/workflows/unit-tests.yml)
# Neural Matrix Factorization
Basic matrix factorization with Keras for missing data imputation.

## Missingness types
### Block-wise
<img src="imgs/missing_values_block_100_split_0.85.png" style="vertical-align: middle; width: 500px; height: 500px;">

### Point-wise
<img src="imgs/missing_values_point_split_0.85.png" style="vertical-align: middle; width: 500px; height: 500px;">

## Neural-MF debugging and experimentation with Tensorboard
Visualization for neural-MF experimentation:
```bash
tensorboard --logdir /path/to/logs/directory
```

