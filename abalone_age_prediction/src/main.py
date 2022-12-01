from __future__ import annotations
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


url = "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv"

def load_csv(url):
    return pd.read_csv(url, 
                        names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
                               "Viscera weight", "Shell weight", "Age"])
def split_dataset(df):
    df_len = int(len(df)*0.8)
    df_train = df[:df_len]
    df_val = df[df_len:]
    return df_train, df_val




class Abalone_dataset:
    def __init__(self, df,  predict_col_name):
        
        self.df = df
        # self.df_train, self.df_val = self.split_dataframe(split_frac)
        self.predict_col_name = predict_col_name 

    def drop_predicting_col(self, df):
        df = df.copy()
        labels = df.pop(self.predict_col_name)
        return df, labels

    def split_dataframe(self, frac):
        df_val = self.df.sample(frac=frac, random_state=np.random.randint(low=1, high=2000+1))
        df_train = self.df.drop(df_val.index)
        return df_train, df_val

    def dataframe_to_dataset(self, df, inputs=None):
        df, labels = self.drop_predicting_col(df)
        if inputs is not None:
            df = df.loc[:, inputs]
        ds = tf.data.Dataset.from_tensor_slices((df, labels))
        return ds.shuffle(len(df))

class Abalone_model:
    BATCH_SIZE = 32
    def __init__(self, train, val, input_shape):
        self.norm = tf.keras.layers.Normalization(axis=-1)
        self.train_ds = train.batch(self.BATCH_SIZE)
        self.val_ds = val.batch(self.BATCH_SIZE)
        self.input_shape = input_shape
        # self.feature_ds()

    #* applies numerical normalization
    def feature_ds(self):
        feature_train = self.train_ds.map(lambda x, y: x)
        self.norm.adapt(feature_train)


    def set_model(self):
        model = tf.keras.Sequential([])
        model.add(tf.keras.layers.InputLayer((self.input_shape,), batch_size=self.BATCH_SIZE))
        model.add(self.norm)
        model.add(tf.keras.layers.Dense(512))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(256))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(64))
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss = tf.keras.losses.MeanAbsolutePercentageError(),
                      optimizer = tf.keras.optimizers.Adam(0.1)
                      )
        # model(self.train_ds)
        model.summary()
        model.fit(self.train_ds, validation_data=self.val_ds, epochs=30)
        return model

if __name__ == "__main__":

    df = load_csv(url)
    ad_age = Abalone_dataset(df, "Age")
    inputs = ["Length", "Diameter", "Height", "Whole weight"]
    train_df, val_df = ad_age.split_dataframe(0.2)
    train_ds_len = ad_age.dataframe_to_dataset(train_df, inputs)
    val_ds_len = ad_age.dataframe_to_dataset(val_df, inputs)

    am = Abalone_model(train_ds_len, val_ds_len, len(inputs))
    model = am.set_model()