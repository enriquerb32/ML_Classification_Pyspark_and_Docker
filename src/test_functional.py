# test_functional.py
import pytest
from pyspark.sql import SparkSession
from src import pyspark_func as psf  # Assuming your PySpark functions are in a separate module

@pytest.fixture
def spark_session():
    return SparkSession.builder.appName('test_healthcare_pyspark').getOrCreate()

def test_training(spark_session):
    # Assuming you have a test DataFrame
    test_data = [[1, 2, 3], [4, 5, 6]]
    test_df = spark_session.createDataFrame(test_data, schema=['col1', 'col2', 'col3'])
    train_df, test_df = psf.prepare_dataset_spark(spark_session, test_df)
    accuracy = psf.training(spark_session, train_df, test_df)
    assert accuracy >= 0.0 and accuracy <= 1.0
