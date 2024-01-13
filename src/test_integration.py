# test_integration.py
import pytest
from pyspark.sql import SparkSession
from src import pyspark_func as psf
from src import scikit_func as sf

@pytest.fixture
def spark_session():
    return SparkSession.builder.appName('test_healthcare_pyspark').getOrCreate()

def test_integration(spark_session):
    test_data = [[1, 2, 3], [4, 5, 6]]
    test_df = spark_session.createDataFrame(test_data, schema=['col1', 'col2', 'col3'])
    
    # Load data using scikit_func
    sf_df = sf.load_data()
    dataframe = spark_session.createDataFrame(sf_df).drop("id")

    # Prepare and train using PySpark functions
    train_df, test_df = psf.prepare_dataset_spark(spark_session, test_df)
    accuracy = psf.training(spark_session, train_df, test_df)

    assert accuracy >= 0.0 and accuracy <= 1.0
