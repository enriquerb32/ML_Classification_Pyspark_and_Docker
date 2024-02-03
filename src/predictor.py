import pandas as pd
import pyspark_func as psf
import scikit_func as sf

# Load your data
data = pd.read_csv("resource/cardio_train.csv", sep=';')

# Create a new row of data (replace this with your own data)
new_data = pd.DataFrame({
    'age': [66],
    'gender': [2],
    'height': [176],
    'weight': [59],
    'ap_hi': [120],
    'ap_lo': [80],
    'cholesterol': [2],
    'gluc': [1],
    'smoke': [0],
    'alco': [0],
    'active': [1]
})

# Assuming 'model' is your trained scikit-learn model
# Assuming 'pyspark_model' is your trained PySpark model

# Preprocess the new data (replace X_new with your feature names)
X_new = new_data[['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']]
X_new_scaled = sf.scalar.transform(X_new)

# Use scikit-learn model for prediction
prediction_scikit = sf.model.predict(X_new_scaled)

# Use PySpark model for prediction
spark_new_data = psf.spark.createDataFrame(new_data)
new_data_vectorized = psf.assembler.transform(spark_new_data).select("features")
prediction_pyspark = psf.pyspark_model.transform(new_data_vectorized).select("prediction").collect()[0][0]

print(f"Scikit-learn Prediction: {prediction_scikit}")
print(f"PySpark Prediction: {prediction_pyspark}")
