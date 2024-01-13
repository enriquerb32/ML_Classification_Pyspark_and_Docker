# test_unit.py
import pytest
from pyspark.sql import SparkSession
from src import scikit_func as sf

@pytest.fixture
def spark_session():
    return SparkSession.builder.appName('test_healthcare_pyspark').getOrCreate()

def test_load_data(spark_session):
    df = sf.load_data()
    for row in df.collect():
        assert isinstance(row, dict)
    assert df.count() > 0

def test_data_prep(spark_session):
    test_data = [[{'col1': 1, 'col2': 2}, {'col1': 4, 'col2': 5}], ...]
    test_df = spark_session.createDataFrame(test_data)
    prepared_df = sf.data_prep(test_df)
    assert prepared_df.count() > 0
