import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
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
    return [(50, 'M', 170, 80, 120, 80, 1, 1, 0, 0, 1, 1),
        (55, 'F', 160, 70, 140, 90, 3, 3, 1, 1, 1, 0),
        (60, 'F', 160, 70, 140, 90, 3, 3, 1, 1, 1, 1),
        (25, 'F', 160, 70, 140, 90, 3, 3, 1, 1, 1, 0),
        (75, 'F', 160, 70, 140, 90, 3, 3, 1, 1, 1, 1),
        (51, 'F', 160, 70, 140, 90, 3, 3, 1, 1, 1, 0),
        (50, 'F', 160, 70, 140, 90, 3, 3, 1, 1, 1, 1),
        (85, 'F', 160, 70, 140, 90, 3, 3, 1, 1, 1, 1),
        (15, 'F', 160, 70, 140, 90, 3, 3, 1, 1, 1, 0),
        (25, 'F', 160, 70, 140, 90, 3, 3, 1, 1, 1, 0)]
    
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
