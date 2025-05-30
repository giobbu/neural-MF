[![Python Tests](https://github.com/giobbu/neural-MF/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/giobbu/neural-MF/actions/workflows/unit-tests.yml)
# Neural Matrix Factorization
Basic matrix factorization with Keras for missing data imputation.

## Missingness types
```python
# Simulate missingness in the DataFrame
training_df, validation_df = simulate_missingness(df,
    split_ratio=params.SPLIT_RATIO,
    block_missingness=params.BLOCK_MISSINGNESS,
    block_size=params.BLOCK_SIZE
)
```
### Block-wise
<img src="imgs/missing_values_block_100_split_0.85.png" style="vertical-align: middle; width: 500px; height: 550px;">

### Point-wise
<img src="imgs/missing_values_point_split_0.85.png" style="vertical-align: middle; width: 500px; height: 550px;">

## Build model architecture and inspect it in Netron
Build Neural-MF and save it as `model.keras`. Visualize layers and architecture in [Netron](https://netron.app/).
```python
# Initialize and build the model
neural_matrix_factor = NeuralMF(training_df, validation_df, K=params.LATENT_DIM)
neural_matrix_factor.build_model(learning_rate=params.LR,
                        reg=params.REG,
                        loss=params.LOSS,
                        metrics=params.METRICS
                        save_path="save/model.keras"  # Path to save the model
                        )
```

<img src="imgs/neural-mf.png" style="vertical-align: middle; width: 500px; height: 400px;">

## Train, optimize and debug with Tensorboard
```python
# Train the model
_ = neural_matrix_factor.train(epochs=params.EPOCHS,
                        batch_size=params.BATCH_SIZE
                        )
```
Visualize experimentation in [Tensorboard](https://www.tensorflow.org/tensorboard):

```bash
tensorboard --logdir /path/to/logs/directory
```

