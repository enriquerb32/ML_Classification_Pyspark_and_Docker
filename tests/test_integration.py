import pytest
from pyspark.sql import SparkSession
import src.pyspark_func as psf
import src.scikit_func as sf

@pytest.fixture
def spark_session():
    spark = SparkSession.builder.appName('test_healthcare_pyspark').getOrCreate()
    yield spark
    spark.stop()

def test_integration(spark_session):
    try:
        # Load data using scikit_func
        sf_df = sf.load_data()

        # Prepare and train using PySpark functions
        train_df, test_df = psf.prepare_dataset_spark(spark_session, sf_df)
        accuracy, _, _, _, _, _, _ = psf.training(spark_session, 'Decision Tree', train_df, test_df)

        assert accuracy >= 0.0 and accuracy <= 1.0

    except Exception as e:
        pytest.fail(f"Integration test failed: {e}")
