import pandas as pd
from os import listdir
from os.path import isfile, join

onlyfiles = [f for f in listdir('./') if isfile(join('./', f))]

metrics_df = pd.read_csv("metrics.csv")

def test_data_zip():
    assert "data.zip" not in onlyfiles

def test_model_h5():
    assert "model.h5" not in onlyfiles

def test_accuracy():
    assert metrics_df['val_accuracy'].max() > 0.7

def test_classwise_accuracy():
    assert metrics_df['cats_acuracy'].max() > 0.7 and metrics_df['dogs_acuracy'].max() > 0.7
