import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.layers import Input, Embedding, Dot, Add, Flatten
from keras.regularizers import l2

import tensorflow as tf
import datetime
import hashlib
import os


class NeuralALS:
    """
    Neural Alternating Least Squares (Neural ALS) for matrix factorization.
    """
    def __init__(self, training_df, validation_df, K=10):
        assert isinstance(training_df, pd.DataFrame) and isinstance(validation_df, pd.DataFrame), \
            "training_df and validation_df must be pandas DataFrames."
        self.training_df = training_df
        self.validation_df = validation_df
        self.K = K
        #self.mu = training_df.OBS.mean()
        self.N = training_df.rowId.max() + 1  # number of observations
        self.M = training_df.columnId.max() + 1  # number of variables
        self.model = None

    def build_model(self, loss='mse', metrics=['mse'], learning_rate=0.01, reg=0.001):
        """
        Build the matrix factorization model using Keras.
        """
        try:
            u = Input(shape=(1,))
            p = Input(shape=(1,))
            u_embedding = Embedding(self.N, self.K, 
                                    embeddings_regularizer=l2(reg))(u)
            p_embedding = Embedding(self.M, self.K, 
                                    embeddings_regularizer=l2(reg))(p)
            u_bias = Embedding(self.N, 1, 
                               embeddings_regularizer=l2(reg))(u)
            p_bias = Embedding(self.M, 1, 
                               embeddings_regularizer=l2(reg))(p)
            x = Dot(axes=2)([u_embedding, p_embedding])  # x = u_embedding * p_embedding
            x = Add()([x, u_bias, p_bias])  # x = x + u_bias + p_bias
            x = Flatten()(x)
            self.model = Model(inputs=[u, p], outputs=x)
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
            self.model.compile(loss=loss, optimizer=opt, metrics=metrics)
        except Exception as e:
            print(f"Error building model: {e}")

    def train(self, epochs=5, batch_size=128, path_save_exp="logs/fit/"):
        """
        Train the matrix factorization model using the training data.
        """
        try:
            # if path folder does not exist, create it
            if not os.path.exists(path_save_exp):
                os.makedirs(path_save_exp)
            key = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = path_save_exp + hashlib.md5(key.encode()).hexdigest()
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            self.history = self.model.fit(x=[self.training_df.rowId.values, self.training_df.columnId.values],
                                        y=self.training_df.OBS.values, # - self.mu,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        validation_data=(
                                            [self.validation_df.rowId.values, self.validation_df.columnId.values],
                                            self.validation_df.OBS.values # - self.mu
                                            ),
                                            callbacks=[tensorboard_callback]
                                        )
            return self.history
        except Exception as e:
            print(f"Error during training: {e}")

    def fill(self, test_df):
        """
        Predict the validation data using the trained model.
        """
        try:
            predictions = self.model.predict([test_df.rowId.values, test_df.columnId.values])# + self.mu
            df_predictions = self.create_df(predictions, test_df)
            return df_predictions
        except Exception as e:
            print(f"Error during validation prediction: {e}")

    def create_df(self, predictions: np.ndarray, validation_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert predictions to DataFrame.
        """
        predictions = pd.Series(predictions.flatten())
        df_predictions = pd.DataFrame(predictions)
        df_predictions['rowId'] = validation_df['rowId'].values
        df_predictions['columnId'] = validation_df['columnId'].values
        df_predictions.columns = ['predictions', 'rowId', 'columnId']
        return df_predictions
    
    def plot_losses(self):
        """
        Plot the training and validation losses.
        """
        plt.plot(self.history.history['loss'], label="train loss")
        plt.plot(self.history.history['val_loss'], label="test loss")
        plt.legend()
        plt.show()

    def plot_metric(self):
        """
        Plot the training and validation MSE.
        """
        plt.plot(self.history.history['mse'], label="train mse")
        plt.plot(self.history.history['val_mse'], label="test mse")
        plt.legend()
        plt.show()


