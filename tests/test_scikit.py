import pytest
import pandas as pd
from src.scikit_func import _load_data, _prepare_dataset, _get_sidebar_classifier, _trigger_classifier

# Fixture to load sample data
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'age': [30, 40, 50, 60, 30, 40, 50, 60, 30, 40, 50, 60],
        'gender': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'height': [160, 170, 165, 175, 160, 170, 165, 175, 160, 170, 165, 175],
        'weight': [60, 70, 65, 75, 70, 65, 75, 70, 65, 75, 70, 65],
        'ap_hi': [120, 130, 140, 150, 120, 130, 140, 150, 120, 130, 140, 150],
        'ap_lo': [80, 90, 85, 95, 90, 85, 95, 90, 85, 95, 90, 100],
        'cholesterol': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        'gluc': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        'smoke': [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        'alco': [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        'active': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'cardio': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })

def test_load_data():
    # Check if data is loaded as expected
    df = _load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_prepare_dataset(sample_data):
    # Check if dataset is prepared correctly
    X_train, X_test, y_train, y_test = _prepare_dataset(sample_data)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

def test_get_sidebar_classifier():
    # Check if classifiers are returned correctly
    classifiers = _get_sidebar_classifier()
    assert isinstance(classifiers, tuple)
    assert 'Decision Tree' in classifiers
    assert 'SVM' in classifiers
    assert 'Random Forest' in classifiers
    assert 'XGBoost' in classifiers

def test_trigger_classifier(sample_data):

    params = dict()
    params['criterion'] = 'gini'
    params['max_features'] = 'sqrt'
    params['max_depth'] = 1
    params['min_samples_split'] = 0.1

    X_train, X_test, y_train, y_test = _prepare_dataset(sample_data)

    # Check if trigger_classifier returns expected results for Decision Tree
    accuracy, _, _, _, _, _, _, _ = _trigger_classifier('Decision Tree', params, X_train, X_test, y_train, y_test)
    assert accuracy >= 0
