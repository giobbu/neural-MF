[![Python Tests](https://github.com/giobbu/neural-MF/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/giobbu/neural-MF/actions/workflows/unit-tests.yml)
# Neural Matrix Factorization
Basic matrix factorization with Keras for missing data imputation.

## Missingness types
### Block-wise
<img src="imgs/missing_values_block_100_split_0.85.png" style="vertical-align: middle; width: 500px; height: 550px;">

### Point-wise
<img src="imgs/missing_values_point_split_0.85.png" style="vertical-align: middle; width: 500px; height: 550px;">

## Model Architecture with Netron
Inspect model structure after saving in [Netron](https://netron.app/)

<img src="imgs/neural-mf.png" style="vertical-align: middle; width: 500px; height: 400px;">

## Debugging and experimentation with Tensorboard
Visualization for neural-MF experimentation in [Tensorboard](https://www.tensorflow.org/tensorboard):

```bash
tensorboard --logdir /path/to/logs/directory
```

