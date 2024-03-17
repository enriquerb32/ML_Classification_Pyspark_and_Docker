import streamlit as st

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