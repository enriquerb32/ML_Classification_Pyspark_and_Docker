import pytest
import pandas as pd
from pyspark.sql import SparkSession
from src.pyspark_func import _training, _clean_dataset, _remove_outliers, _attribute_combination, _prepare_train_test

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
def sample_data(spark_session):
    pandas_sample = pd.DataFrame({
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

    spark_sample = spark_session.createDataFrame(pandas_sample)

    return spark_sample
    
def test_prepare_dataset_spark(spark_session, sample_data):
    cols_to_impute = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']

    # Load your data and preprocess it
    cleaned_df = _clean_dataset(sample_data, cols_to_impute)
    processed_df = _attribute_combination(cleaned_df)

    # Prepare training and test data
    trainingData, testData, _, _, _, _ = _prepare_train_test(spark_session, processed_df)

    assert trainingData.count() > 0
    assert testData.count() > 0
    return trainingData, testData

def test_training(spark_session, sample_data):
    train_data, test_data = test_prepare_dataset_spark(spark_session, sample_data)
    accuracy, fpr, tpr, auc, f1, _, _ = _training(spark_session, 'Decision Tree', train_data, test_data)
    assert accuracy >= 0
    assert fpr >= 0
    assert tpr >= 0
    assert auc >= 0
    assert f1 >= 0
