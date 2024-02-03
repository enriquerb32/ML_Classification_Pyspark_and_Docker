from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.evaluation import BinaryClassificationMetrics
#from pyspark.ml.evaluation import BinaryClassificationEvaluator

import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items

def get_spark_session():
    return SparkSession.builder.appName('healthcare_pyspark') \
          .config("spark.pyspark.python", "python3") \
          .config("spark.pyspark.driver.python", "python3") \
          .getOrCreate()

def prepare_dataset_spark(spark, pd_data):
    dataframe = spark.createDataFrame(pd_data).drop("id")
    inputCols = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    assembler = VectorAssembler(inputCols=inputCols, outputCol='features')
    df_temp = assembler.transform(dataframe).select("features", "cardio")

    # Cast the 'cardio' column to DoubleType
    df_temp = df_temp.withColumn("cardio", df_temp["cardio"].cast(DoubleType()))

    (trainingData, testData) = df_temp.randomSplit([0.7, 0.3])
    return trainingData, testData

def get_model(classifier, params):
    if classifier == 'Decision Tree':
        return DecisionTreeClassifier(labelCol="cardio", featuresCol="features")
    elif classifier == 'Random Forest':
        return RandomForestClassifier(labelCol="cardio", featuresCol="features")

def training(spark, classifier, train_df, test_df):
    pyspark_classifier = get_model(classifier, None)
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