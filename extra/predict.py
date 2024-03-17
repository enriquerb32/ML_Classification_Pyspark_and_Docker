import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, round, lit, udf
from pyspark.ml.stat import Correlation
from pyspark.sql.types import DoubleType
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pyspark_func as psf

def _predictor(spark, df):
    # Create a new row of data (replace this with your own data)
    df_new = spark.createDataFrame([
        (40, 1, 163, 90, 135, 85, 1, 1, 0, 1, 0)
    ], ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"])

    # Calculate BMI and MAP
    df_new = df_new.withColumn("bmi", df_new['weight'] / (df_new['height'] ** 2))
    df_new = df_new.withColumn("map", (2 * df_new['ap_lo'] + df_new['ap_hi']) / 3)

    # Calculate a fictitious "cardio" column to match the columns of the training dataset
    df_new = df_new.withColumn("cardio", lit(9999))

    # Drop unnecessary columns
    columns_to_drop = ['weight', 'height', 'ap_hi', 'ap_lo']
    df_new = df_new.drop(*columns_to_drop)

    # Generate a range of ages from the imputed age to 100 with a step of 1 between elements
    agerange = list(range(40, 66))

    # Replicate new data for each age
    df_replicated = df_new.crossJoin(spark.range(len(agerange)).withColumnRenamed("id", "idx"))
    df_replicated = df_replicated.withColumn("age", df_replicated.age + df_replicated.idx)
    df_replicated = df_replicated.drop("idx")

    # Assemble features into a single vector column and perform the preprocessing
    feature_columns = ['age', 'gender', 'bmi', 'map', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df_replicated = assembler.transform(df_replicated).select('features', 'cardio')
    df_replicated = df_replicated.withColumn('cardio', df_replicated['cardio'].cast(DoubleType()))

    # Define classifiers
    rf = RandomForestClassifier(labelCol="cardio", featuresCol="features")

    # Define hyperparameter grid for Random Forest
    paramGrid_rf = ParamGridBuilder() \
        .addGrid(rf.featureSubsetStrategy, ['auto', 'sqrt']) \
        .addGrid(rf.maxDepth, list(range(2, 25))) \
        .addGrid(rf.numTrees, [int(500 * np.random.power(1)) for _ in range(3)]) \
        .addGrid(rf.impurity, ['gini', 'entropy']) \
        .build()

    # Define evaluator
    evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC").setLabelCol("cardio")

    # Define cross-validator
    cv = CrossValidator(estimator=rf,
                        estimatorParamMaps=paramGrid_rf,
                        evaluator=evaluator,
                        numFolds=3)

    # Train and evaluate models using the combined pipeline
    cv_models = cv.fit(df)

    # Make predictions
    predictions = cv_models.transform(df_replicated)

    # Define a UDF to extract the probability of the positive class
    extract_positive_prob = udf(lambda probability: float(probability[1]), DoubleType())

    # Apply the UDF to the "probability" column to extract the probability of the positive class
    predictions = predictions.withColumn("probability_cardio", extract_positive_prob("probability"))

    # Collect predictions into a Pandas DataFrame
    predictions_pd = predictions.select('age', 'probability_cardio').toPandas()

    return predictions_pd
