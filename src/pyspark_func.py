from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

def get_spark_session():
    return SparkSession.builder.appName('healthcare_pyspark') \
        .config("spark.pyspark.python", "python3") \
        .config("spark.pyspark.driver.python", "python3") \
        .getOrCreate()

def prepare_dataset_spark(spark, pd_data):
    dataframe = spark.createDataFrame(pd_data).drop("id")
    inputCols = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco',
                 'active']
    assembler = VectorAssembler(inputCols=inputCols, outputCol='features')
    df_temp = assembler.transform(dataframe).select("features", "cardio")

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
    
    # Extract probability values
    y_true = [float(row['cardio']) for row in rf_predictions.collect()]
    y_prob = [float(row['probability'][1]) for row in rf_predictions.select('probability').collect()]
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    # Calculate feature importances
    feature_importances = model.featureImportances.toArray().tolist()

    # Calculate confusion matrix
    y_pred = [float(row['prediction']) for row in rf_predictions.collect()]
    confusion_mat = confusion_matrix(y_true, y_pred)

    return model, fpr, tpr, auc, feature_importances, confusion_mat
