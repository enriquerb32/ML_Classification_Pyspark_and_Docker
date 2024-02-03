import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from src.pyspark_func import prepare_dataset_spark, training

# Fixture to create a SparkSession
@pytest.fixture(scope="session")
def spark_session():
    spark = SparkSession.builder \
        .appName("test") \
        .master("local[2]") \
        .getOrCreate()
    yield spark
    spark.stop()

# Fixture to create a sample DataFrame for testing
@pytest.fixture(scope="session")
def sample_data(spark_session):
    schema = StructType([
        StructField("age", IntegerType(), True),
        StructField("gender", StringType(), True),
        StructField("height", IntegerType(), True),
        StructField("weight", IntegerType(), True),
        StructField("ap_hi", IntegerType(), True),
        StructField("ap_lo", IntegerType(), True),
        StructField("cholesterol", IntegerType(), True),
        StructField("gluc", IntegerType(), True),
        StructField("smoke", IntegerType(), True),
        StructField("alco", IntegerType(), True),
        StructField("active", IntegerType(), True),
        StructField("cardio", IntegerType(), True)
    ])
    data = [
        (50, 'M', 170, 80, 120, 80, 1, 1, 0, 0, 1, 1),
        (55, 'F', 160, 70, 140, 90, 3, 3, 1, 1, 1, 0)
    ]
    return spark_session.createDataFrame(data, schema)

def test_prepare_dataset_spark(spark_session, sample_data):
    train_data, test_data = prepare_dataset_spark(spark_session, sample_data)
    assert train_data.count() > 0
    assert test_data.count() > 0

def test_training(spark_session, sample_data):
    train_data, test_data = prepare_dataset_spark(spark_session, sample_data)
    accuracy, fpr, tpr, auc, f1, _, _ = training(spark_session, 'Decision Tree', train_data, test_data)
    assert accuracy >= 0
    assert fpr >= 0
    assert tpr >= 0
    assert auc >= 0
    assert f1 >= 0