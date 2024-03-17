import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from scikit_func import load_data, prepare_dataset, get_sidebar_classifier, trigger_classifier
from pyspark_func import get_spark_session, _data_io, _clean_dataset, _remove_outliers, _attribute_combination,_prepare_train_test, _get_model, _training, _correlation_matrix, _predictor
from streamlit_func import add_parameter_ui

def main():
    spark = get_spark_session()

    # Load your data and preprocess it
    original_df, cols_to_impute = _data_io(spark)
    cleaned_df = _clean_dataset(original_df, cols_to_impute)
    outliers_removed_df = _remove_outliers(cleaned_df)
    processed_df = _attribute_combination(outliers_removed_df)

    # Prepare training and test data
    pandas_df = processed_df.toPandas()
    trainingData, testData, X_train, X_test, y_train, y_test = _prepare_train_test(spark, processed_df)

    st.title('Streamlit Healthcare ML Data App')
    st.subheader('Streamlit Healthcare example') 
    st.markdown("**Cardiovascular Disease dataset by Kaggle**")
    st.markdown('''
        This is the source of the dataset and problem statement 
        [Kaggle Link](https://www.kaggle.com/sulianova/cardiovascular-disease-dataset)  
        The dataset consists of 70,000 records of patient data, 11 features + target
        Running the same data over ___Scikit Learn & Pyspark ML___ - A simple Comparison and Selection  
    ''')

    #st.dataframe(data=processed_df.head(20), height=200)
    #st.write('Shape of dataset:', X_train.shape)
    #st.write('number of classes:', len(np.unique(y_test)))

    st.subheader("Correlation Matrix")
    correlation_matrix_plotly = px.imshow(_correlation_matrix(spark, processed_df.drop('cardio')), color_continuous_scale='Viridis')
    st.plotly_chart(correlation_matrix_plotly, use_container_width=True)

    col_plot1, col_plot2 = st.columns(2)

    with col_plot1:
        st.subheader('Mean Arterial Pressure (MAP) Histogram')
        fig, ax = plt.subplots()
        sns.histplot(pandas_df['map'], palette="Set2", ax=ax)
        ax.set_xlabel('MAP')
        st.pyplot(fig)

        st.subheader('Age Histogram')
        fig, ax = plt.subplots()
        sns.histplot(pandas_df['age'], palette="Set2", ax=ax)
        ax.set_xlabel('Age')
        st.pyplot(fig)

        st.subheader('Absence/Presence of CVD over Age')
        fig, ax = plt.subplots()
        sns.countplot(x='age', hue='cardio', data=pandas_df, palette="Set2", ax=ax)
        ax.set_xlabel('Year')
        ax.set_ylabel('Count')
        st.pyplot(fig)

    with col_plot2:
        st.subheader('Body Mass Index (BMI) Histogram')
        fig, ax = plt.subplots()
        sns.histplot(pandas_df['bmi'], palette="Set2", ax=ax)
        ax.set_xlabel('BMI')
        st.pyplot(fig)

        st.subheader('BMI vs. MAP Scatterplot')
        fig, ax = plt.subplots()
        sns.scatterplot(x='bmi', y='map', data=pandas_df, palette="Set2", ax=ax)
        ax.set_xlabel('BMI')
        ax.set_ylabel('MAP')
        st.pyplot(fig)

        st.subheader('Distribution of attributes over people with CVD')
        fig, ax = plt.subplots()
        df_categorical = pandas_df.loc[:, ['cholesterol', 'gluc', 'smoke', 'alco', 'active']]
        sns.countplot(x="variable", hue="value", data=pd.melt(df_categorical), ax=ax)
        ax.set_xlabel('Variable')
        ax.set_ylabel('Count')
        st.pyplot(fig)

    # Sidebar: Classifier Selection and Parameters
    st.sidebar.title("Classifier Selection")
    classifier_type = st.sidebar.selectbox("Select Classifier", get_sidebar_classifier())
    params = add_parameter_ui(classifier_type)

    # Trigger classifier and retrieve results
    sk_classifier_results = trigger_classifier(classifier_type, params, X_train, X_test, y_train, y_test)
    sk_accuracy, sk_feature_importances, sk_fpr, sk_tpr, sk_thresholds, sk_f1, sk_auc, sk_confusion_mat = sk_classifier_results

    # Trigger classifier and retrieve results for PySpark
    if classifier_type in ['Decision Tree', 'Random Forest']:
        classifier_results = _training(spark, classifier_type, trainingData, testData)
        accuracy, fpr, tpr, auc, f1, feature_importances, confusion_mat = classifier_results
    
    col_plot3, col_plot4 = st.columns(2)

    # Scikit Plots
    with col_plot3:
        # Left half
        col_plot3.header('''
        __Scikit Learn ML__  
        ![scikit](https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png)''')
        st.write(f"Accuracy: {sk_accuracy}")
        st.write(f"F1 Score: {sk_f1}")
        st.write(f"AUC: {sk_auc}")      

        if feature_importances is not None:
            st.subheader("Feature Importances")
            features_data = pd.DataFrame({'Feature': X_test.columns, 'Importance': sk_feature_importances.tolist()})
            st.bar_chart(features_data, x='Feature', y='Importance')

        #st.subheader("ROC Curve")
        #roc_data = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        #st.area_chart(roc_data, x="FPR", y="TPR")

        st.subheader("Confusion Matrix")
        confusion_matrix_plotly = ff.create_annotated_heatmap(sk_confusion_mat, colorscale='Viridis')
        st.plotly_chart(confusion_matrix_plotly, use_container_width=True)

     # PySpark Plots   
    with col_plot4:
        col_plot4.header('''
        __PySpark Learn ML__  
        ![pyspark](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Apache_Spark_logo.svg/120px-Apache_Spark_logo.svg.png)
        ''')
        if classifier_type in ['Decision Tree', 'Random Forest']:
            st.write(f"Accuracy: {accuracy}")
            st.write(f"F1 Score: {f1}")
            st.write(f"AUC: {auc}")

            if feature_importances is not None:
                st.subheader("Feature Importances")
                pyspark_features_data = pd.DataFrame({'Feature': X_test.columns, 'Importance': feature_importances})
                st.bar_chart(pyspark_features_data, x='Feature', y='Importance')

            #st.subheader("ROC Curve")
            #pyspark_roc_data = pd.DataFrame({"FPR": pyspark_fpr, "TPR": pyspark_tpr})
            #st.area_chart(pyspark_roc_data, x="FPR", y="TPR")

            st.subheader("Confusion Matrix")
            pyspark_confusion_matrix_plotly = ff.create_annotated_heatmap(confusion_mat, colorscale='Viridis')
            st.plotly_chart(pyspark_confusion_matrix_plotly, use_container_width=True)

    # Retrieve the predictor function so to obtain the probability of CVD between the 40 and 65 years old person
    # We are using PySpark's Random Forest with an automatic hyperparameter tuning
    predictions = _predictor(spark, trainingData)

    # Evaluate predictions over patient's age
    st.subheader('Predictions on CVD risk over age of the sample patient')
    fig, ax = plt.subplots()
    sns.scatterplot(x='age', y='probability_cardio', data=predictions, palette="Set2", ax=ax)
    ax.set_xlabel('Age')
    ax.set_ylabel('Percentual CVD Risk')
    st.pyplot(fig)

if __name__ == "__main__":
    main()
