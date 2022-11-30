import pytest
from src.main import load_csv, split_dataset, drop_predicting_col

url = "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv"

@pytest.fixture
def csv():
    return load_csv(url)

def test_split_dataset(csv):
    train, val = split_dataset(csv)
    print(train["Length"])
    print(val.head())
    assert 0

def test_drop_predicting_col(csv):
    train, labels = drop_predicting_col(csv, "Age")
    print(train.head(), labels.head())
    assert 0
