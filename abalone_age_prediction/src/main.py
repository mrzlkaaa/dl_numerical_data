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

def drop_predicting_col(df, col_name):
    labels = df.pop(col_name)
    return df, labels


# class Model:
    # def __init__(self):


if __name__ == "__main__":

    df = load_csv(url)
    df_train, df_test = split_dataset(df)
    df_train_data, df_train_labels = drop_predicting_col(df_train, "Age")
    df_test_data, df_test_labels = drop_predicting_col(df_test, "Age")

    #! make a class of function to exclude hardcoding
    # norm = tf.keras.layers.Normalization()
    # norm.adapt(df_train_data["Length"].to_numpy())
    # norm_train = norm(df_train_data["Length"].to_numpy())
    # norm_test = norm(df_test_data["Length"].to_numpy())
    # print(df_train_data["Length"], len(norm_train))

    # model = tf.keras.Sequential([])
    # # model.add(norm)
    # model.add(tf.keras.layers.Dense(512, activation="relu"))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(256, activation="relu"))
    # # model.add(tf.keras.layers.Dense(128, activation="relu"))
    # model.add(tf.keras.layers.Dense(1))

    # # model.summary()

    # model.compile(loss = tf.keras.losses.MeanSquaredError(),
    #             optimizer = tf.keras.optimizers.Adam(0.001),
    #             metrics=["accuracy"])

    # model.fit(norm_train, df_train_labels, epochs=25)
    # predicted = model.predict(norm_test)
    # print(predicted.shape)
    # # for i in range(600, len(predicted)):
    # #     print(predicted[i], df_val_labels.iloc[i])
    # plt.scatter(predicted, df_test_labels.to_numpy())
    # plt.savefig("0.png")

