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
from pyspark.ml import pipeline

import numpy as np
import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items

def get_spark_session():
    # Create our basic spark cluster
    return SparkSession.builder.appName('healthcare_pyspark') \
          .config("spark.pyspark.python", "python3") \
          .config("spark.pyspark.driver.python", "python3") \
          .getOrCreate()

def _data_io(spark):
    sdf = spark.read.csv('resource/cardio_train.csv', inferSchema=True, header=True, sep=';')

    sdf = sdf.drop('id')

    col_to_impute = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
    return sdf, col_to_impute

def _clean_dataset(sdf: DataFrame, col_to_impute: list) -> DataFrame:
    sdf = sdf.withColumn('height', round(sdf['height'] / 100, 2))
    sdf = sdf.withColumn('age', round(sdf['age'] / 365.25))
    sdf = sdf.withColumn('cardio', sdf['cardio'].cast(DoubleType()))

    # Use Imputer to fill missing values with median for specified columns
    imputer = Imputer(inputCols=col_to_impute, outputCols=col_to_impute, strategy='median')
    sdf = imputer.fit(sdf).transform(sdf)
    
    return sdf

def _remove_outliers(df):
    # Define quantiles
    quantiles = [0.025, 0.975]

    # Calculate quantiles for each column
    quantile_dict = {col_name: df.approxQuantile(col_name, quantiles, 0.01) for col_name in df.columns}

    # Filter rows based on quantiles for each column
    for col_name, quantile_values in quantile_dict.items():
        lower_quantile = quantile_values[0]
        upper_quantile = quantile_values[1]
    
        df = df.filter(~((col(col_name) < lower_quantile) | (col(col_name) > upper_quantile)))

    return df

def _attribute_combination(df):
    # Calculate BMI and create a new column 'bmi'
    df = df.withColumn('bmi', df['weight'] / (df['height'] ** 2))

    # Calculate MAP and create a new column 'map'
    df = df.withColumn('map', (2 * df['ap_lo'] + df['ap_hi']) / 3)

    # Drop the original columns used for BMI and MAP calculation
    df = df.drop('weight', 'height', 'ap_hi', 'ap_lo')

    return df

def _prepare_train_test(spark, df):
    inputCols = ['age', 'gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi', 'map']
    (trainingData, testData) = df.randomSplit([0.7, 0.3])

    # Convert Spark DataFrames to Pandas DataFrames for SciKit functionalities
    X_train = trainingData.select(inputCols).toPandas()
    X_test = testData.select(inputCols).toPandas()
    y_train = trainingData.select('cardio').toPandas()
    y_test = testData.select('cardio').toPandas()

    assembler = VectorAssembler(inputCols=inputCols, outputCol='features')

    # All the features have transformed into a Dense Vector
    trainingData2 = assembler.transform(trainingData).select('features', 'cardio')
    testData2 = assembler.transform(testData).select('features', 'cardio')

    # Cast the 'cardio' column to DoubleType
    trainingData2 = trainingData2.withColumn('cardio', trainingData2['cardio'].cast(DoubleType()))
    testData2 = testData2.withColumn('cardio', testData2['cardio'].cast(DoubleType()))

    return trainingData2, testData2, X_train, X_test, y_train, y_test

def _get_model(classifier, params):
    if classifier == 'Decision Tree':
        return DecisionTreeClassifier(labelCol="cardio", featuresCol="features")
    elif classifier == 'Random Forest':
        return RandomForestClassifier(labelCol="cardio", featuresCol="features")

def _training(spark, classifier, train_df, test_df):
    pyspark_classifier = _get_model(classifier, None)
    model = pyspark_classifier.fit(train_df)

    # Calculate ROC and AUC
    rf_predictions = model.transform(test_df)

    # y_prob contains the probabilities of the positive class (CVD)
    # y_true contains the true labels (1 for positive, 0 for negative)
    y_true = [float(row['cardio']) for row in rf_predictions.collect()]
    y_prob = [float(row['probability'][1]) for row in rf_predictions.select('probability').collect()]

    predictionAndLabels = spark.sparkContext.parallelize(zip(y_prob, y_true))
    metrics = BinaryClassificationMetrics(predictionAndLabels)

    # Calculate AUC
    auc = metrics.areaUnderROC

    # Calculate feature importances
    feature_importances = model.featureImportances.toArray().tolist()

    # Calculate confusion matrix
    conf_predictionAndLabels = rf_predictions.select("prediction", "cardio").rdd
    conf_metrics = MulticlassMetrics(conf_predictionAndLabels)
    confusion_mat = conf_metrics.confusionMatrix().toArray()

    # Calculate False Positive Rate (FPR) and True Positive Rate (TPR)
    fpr = int(conf_metrics.weightedFalsePositiveRate)
    tpr = int(conf_metrics.weightedTruePositiveRate)

    # Calculate accuracy
    accuracy = conf_metrics.accuracy

    # Calculate F1 score
    f1 = conf_metrics.weightedFMeasure()

    return accuracy, fpr, tpr, auc, f1, feature_importances, confusion_mat

def _correlation_matrix(spark, df):
    # Assemble all feature columns into a single vector column
    feature_cols = df.columns
    vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
    dataset = vector_assembler.transform(df)

    # Calculate the correlation matrix
    pearson_corr = Correlation.corr(dataset, 'features', 'pearson').collect()[0][0]

    # Convert PySpark DataFrame to Pandas DataFrame
    corr_matrix_df = pd.DataFrame(pearson_corr.toArray(), columns=feature_cols, index=feature_cols)

    return corr_matrix_df

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
    df_replicated = assembler.transform(df_replicated).select('age','features', 'cardio')
    df_replicated = df_replicated.withColumn('cardio', df_replicated['cardio'].cast(DoubleType()))

    # Define classifiers
    rf = RandomForestClassifier(labelCol="cardio", featuresCol="features")

    # Train and evaluate models using the training data
    models = rf.fit(df)

    # Make predictions
    predictions = models.transform(df_replicated) 

    # Define a UDF to extract the probability of the positive class
    extract_positive_prob = udf(lambda probability: float(probability[1]), DoubleType())

    # Apply the UDF to the "probability" column to extract the probability of the positive class
    predictions = predictions.withColumn("probability_cardio", extract_positive_prob("probability"))

    # Collect predictions into a Pandas DataFrame
    predictions_pd = predictions.select('age', 'probability_cardio').toPandas()

    return predictions_pd
