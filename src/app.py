import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
from matplotlib.figure import Figure
from scikit_func import load_data, prepare_dataset, get_sidebar_classifier, trigger_classifier
from pyspark_func import get_spark_session, prepare_dataset_spark, training


def add_parameter_ui(clf_name):
    params = dict()

    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C

    elif clf_name == 'XGBoost':
        n_estimators = st.sidebar.slider('n_estimators', 10, 100)
        max_depth = st.sidebar.slider('max_depth', 3, 12)
        learning_rate = st.sidebar.slider('learning_rate', 0.01, 0.1)
        params['n_estimators'] = n_estimators
        params['max_depth'] = max_depth
        params['learning_rate'] = learning_rate

    elif clf_name == 'Decision Tree':
        params['criterion'] = st.sidebar.radio("criterion", ('gini', 'entropy'))
        params['max_features'] = st.sidebar.selectbox("max_features", (None, 'auto', 'sqrt', 'log2'))
        params['max_depth'] = st.sidebar.slider('max_depth', 1, 32)
        params['min_samples_split'] = st.sidebar.slider('min_samples_split', 0.1, 1.0)

    elif clf_name == 'Random Forest':
        num_trees = st.sidebar.slider('numTrees', 10, 100, step=10)
        max_depth = st.sidebar.slider('maxDepth', 3, 12)
        min_info_gain = st.sidebar.slider('minInfoGain', 0.0, 0.5, step=0.01)

        params['numTrees'] = num_trees
        params['maxDepth'] = max_depth
        params['minInfoGain'] = min_info_gain

    return params

# Load data and remove the 'id' column that offers no additional info
data = load_data()
data.drop('id', axis=1, inplace=True)

# Prepare dataset with the specified target column
X_train, X_test, y_train, y_test = prepare_dataset(data)

def main():
    st.title('Streamlit Healthcare ML Data App')
    st.subheader('Streamlit Healthcare example') 
    st.markdown("**Cardiovascular Disease dataset by Kaggle**")
    st.markdown('''
        This is the source of the dataset and problem statement 
        [Kaggle Link](https://www.kaggle.com/sulianova/cardiovascular-disease-dataset)  
        The dataset consists of 70,000 records of patient data, 11 features + target
        Running the same data over ___Scikit Learn & Pyspark ML___ - A simple Comparison and Selection  
    ''')

    st.dataframe(data=data.head(20), height=200)
    st.write('Shape of dataset:', X_train.shape)
    st.write('number of classes:', len(np.unique(y_test)))
    #st.write(f"Accuracy: {accuracy}")

    # Calculate dataframe without the target column because we are calculating 
    # its correlation and we are not using external modules to perform it

    data_notarg = data.copy()
    data_notarg.drop('cardio', axis=1, inplace=True)

    st.subheader("Correlation Matrix")
    correlation_matrix_plotly = px.imshow(data_notarg.corr(), color_continuous_scale='Viridis')
    st.plotly_chart(correlation_matrix_plotly, use_container_width=True)

    col_plot1, col_plot2 = st.columns(2)
    temp_df = data
    with col_plot1:
        st.subheader('Age over Absence/Presence of CVD')
        fig = Figure()
        ax = fig.subplots()
        temp_df['years'] = (temp_df['age'] / 365).round().astype('int')
        sns.countplot(x='years', hue='cardio', data=temp_df, palette="Set2", ax=ax)

        ax.set_xlabel('Year')
        ax.set_ylabel('Count')
        st.pyplot(fig)

    with col_plot2:
        st.subheader('Distribution of attributes over people with CVD')
        fig = Figure()
        ax = fig.subplots()
        df_categorical = temp_df.loc[:, ['cholesterol', 'gluc', 'smoke', 'alco', 'active']]
        sns.countplot(x="variable", hue="value", data=pd.melt(df_categorical), ax=ax)
        ax.set_xlabel('Variable')
        ax.set_ylabel('Count')
        st.pyplot(fig)

    # Sidebar: Classifier Selection and Parameters
    st.sidebar.title("Classifier Selection")
    classifier_type = st.sidebar.selectbox("Select Classifier", get_sidebar_classifier())
    params = add_parameter_ui(classifier_type)

    # Trigger classifier and retrieve results
    classifier_results = trigger_classifier(classifier_type, params, X_train, X_test, y_train, y_test)
    accuracy, feature_importances, fpr, tpr, thresholds, f1, auc, confusion_mat = classifier_results

    # Trigger classifier and retrieve results for PySpark
    if classifier_type in ['Decision Tree', 'Random Forest']:
        spark = get_spark_session()
        spark_train_data, spark_test_data = prepare_dataset_spark(spark, data)
        pyspark_classifier_results = training(spark, classifier_type, spark_train_data, spark_test_data)
        (
            pyspark_accuracy,
            pyspark_fpr,
            pyspark_tpr,
            pyspark_auc,
            pyspark_f1,
            pyspark_feature_importances,
            pyspark_confusion_mat
        ) = pyspark_classifier_results

    
    col_plot3, col_plot4 = st.columns(2)

    # Scikit Plots
    with col_plot3:
        # Left half
        col_plot3.header('''
        __Scikit Learn ML__  
        ![scikit](https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png)''')
        st.write(f"Accuracy: {accuracy}")
        st.write(f"F1 Score: {f1}")
        st.write(f"AUC: {auc}")      

        if feature_importances is not None:
            st.subheader("Feature Importances")
            features_data = pd.DataFrame({'Feature': data.columns[:-2], 'Importance': feature_importances.tolist()})
            st.bar_chart(features_data, x='Feature', y='Importance')


        #st.subheader("ROC Curve")
        #roc_data = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        #st.area_chart(roc_data, x="FPR", y="TPR")

        st.subheader("Confusion Matrix")
        confusion_matrix_plotly = ff.create_annotated_heatmap(confusion_mat, colorscale='Viridis')
        st.plotly_chart(confusion_matrix_plotly, use_container_width=True)


     # PySpark Plots   
    with col_plot4:
        col_plot4.header('''
        __PySpark Learn ML__  
        ![pyspark](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Apache_Spark_logo.svg/120px-Apache_Spark_logo.svg.png)
        ''')
        if classifier_type in ['Decision Tree', 'Random Forest']:
            st.write(f"Accuracy: {pyspark_accuracy}")
            st.write(f"F1 Score: {pyspark_f1}")
            st.write(f"AUC: {pyspark_auc}")

            if pyspark_feature_importances is not None:
                st.subheader("Feature Importances")
                pyspark_features_data = pd.DataFrame({'Feature': data.columns[:-2], 'Importance': pyspark_feature_importances})
                st.bar_chart(pyspark_features_data, x='Feature', y='Importance')

            #st.subheader("ROC Curve")
            #pyspark_roc_data = pd.DataFrame({"FPR": pyspark_fpr, "TPR": pyspark_tpr})
            #st.area_chart(pyspark_roc_data, x="FPR", y="TPR")

            st.subheader("Confusion Matrix")
            pyspark_confusion_matrix_plotly = ff.create_annotated_heatmap(pyspark_confusion_mat, colorscale='Viridis')
            st.plotly_chart(pyspark_confusion_matrix_plotly, use_container_width=True)


if __name__ == "__main__":
    main()
