import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
import pyspark_func as psf
import scikit_func as sf

# Load your data
df_csv = sf.load_data()

# Create a new row of data (replace this with your own data)
df_new = pd.DataFrame({
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
    'active': [1],
    'cardio': [np.nan]
})

# Preprocess the main data
X_train, X_test, y_train, y_test = sf.prepare_dataset(df_csv)

# Preprocess the new data
df = pd.concat([df_csv, df_new], ignore_index=True)
X_df = df.drop(columns=['cardio'])
scalar = MinMaxScaler()
X_df1 = scalar.fit_transform(X_df)
lastrow = X_df1[-1]

# We set the hyperparameters of our prediction
params = dict()
params['criterion'] = 'mse'
params['max_features'] = 'sqrt'
params['max_depth'] = 1
params['min_samples_split'] = 0.1

model = tree.DecisionTreeRegressor()
model = model.fit(X_train, y_train)
result = model.predict([lastrow])

print(f"Scikit-learn Prediction: {result}")
#print(f"PySpark Prediction: {prediction_pyspark}")
