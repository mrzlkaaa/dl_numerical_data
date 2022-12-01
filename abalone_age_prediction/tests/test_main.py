import pytest
from src.main import load_csv, Abalone_dataset, Abalone_model

url = "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv"

@pytest.fixture
def abalone():
    df = load_csv(url)
    return Abalone_dataset(df, "Age")
    
@pytest.fixture
def test_split_dataframe(abalone):
    train, val = abalone.split_dataframe(0.2)
    # print(train.head())
    # print(val.head())
    return (train, val)

def test_dataframe_to_dataset(abalone, test_split_dataframe):
    train, val = test_split_dataframe
    ds_train = abalone.dataframe_to_dataset(train, ["Length"])
    ds_val = abalone.dataframe_to_dataset(val, ["Length"])
    print(ds_train, ds_val)
    assert 0

@pytest.fixture
def df_to_ds(abalone, test_split_dataframe):
    train, val = test_split_dataframe
    ds_train = abalone.dataframe_to_dataset(train, ["Length"])
    ds_val = abalone.dataframe_to_dataset(val, ["Length"])
    return (ds_train, ds_val)

@pytest.fixture
def abalone_model(df_to_ds):
    ds_train, ds_val = df_to_ds
    for x, y in ds_train.take(1):
        print("Input:", x)
        print("Target:", y)
    return Abalone_model(ds_train, ds_val, 1)

def test_feature_ds(abalone_model):
    print(abalone_model.train_ds)
    abalone_model.feature_ds()
    assert 0