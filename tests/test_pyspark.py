import pytest
import pandas as pd
from pyspark.sql import SparkSession
from src.pyspark_func import training, prepare_dataset_spark

# Fixture to create a SparkSession
@pytest.fixture(scope="session")
def spark_session():
    spark = SparkSession.builder \
        .appName("test") \
        .master("local[2]") \
        .getOrCreate()
    yield spark
    spark.stop()

# Fixture to load sample data
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'age': [30, 40, 50, 60, 30, 40, 50, 60, 30, 40, 50, 60],
        'gender': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'height': [160, 170, 165, 175, 160, 170, 165, 175, 160, 170, 165, 175],
        'weight': [60, 70, 65, 75, 70, 65, 75, 70, 65, 75, 70, 65],
        'ap_hi': [120, 130, 140, 150, 120, 130, 140, 150, 120, 130, 140, 150],
        'ap_lo': [80, 90, 85, 95, 90, 85, 95, 90, 85, 95, 90, 100],
        'cholesterol': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        'gluc': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        'smoke': [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        'alco': [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        'active': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'cardio': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    
def test_prepare_dataset_spark(spark_session, sample_data):  
    trainingData, testData = prepare_dataset_spark(spark_session, sample_data)
    assert trainingData.count() > 0
    assert testData.count() > 0
    return trainingData, testData

def test_training(spark_session, sample_data):
    train_data, test_data = prepare_dataset_spark(spark_session, sample_data)
    accuracy, fpr, tpr, auc, f1, _, _ = training(spark_session, 'Decision Tree', train_data, test_data)
    assert accuracy >= 0
    assert fpr >= 0
    assert tpr >= 0
    assert auc >= 0
    assert f1 >= 0
