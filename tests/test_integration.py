import pytest
from pyspark.sql import SparkSession
import src.pyspark_func as psf

@pytest.fixture
def spark_session():
    spark = SparkSession.builder.appName('test_healthcare_pyspark').getOrCreate()
    yield spark
    spark.stop()

def test_integration(spark_session):
    try:
        # Load your data and preprocess it
        original_df, cols_to_impute = psf._data_io(spark_session)
        cleaned_df = psf._clean_dataset(original_df, cols_to_impute)
        outliers_removed_df = psf._remove_outliers(cleaned_df)
        processed_df = psf._attribute_combination(outliers_removed_df)

        # Prepare and train using PySpark functions
        train_df, test_df, _, _, _, _ = psf._prepare_train_test(spark_session, processed_df)
        accuracy, _, _, _, _, _, _ = psf._training(spark_session, 'Decision Tree', train_df, test_df)

        assert accuracy >= 0.0 and accuracy <= 1.0

    except Exception as e:
        pytest.fail(f"Integration test failed: {e}")
