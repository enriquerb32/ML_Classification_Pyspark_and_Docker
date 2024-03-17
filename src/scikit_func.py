import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, f1_score


def _load_data():
    return pd.read_csv("resource/cardio_train.csv", sep=';')


def _get_sidebar_classifier():
    return 'Decision Tree', 'SVM', 'Random Forest', 'XGBoost'


def _prepare_dataset(dataframe):
    X = dataframe.drop(columns=['cardio'])
    y = dataframe['cardio']
    scalar = MinMaxScaler()
    x_scaled = scalar.fit_transform(X)
    return train_test_split(x_scaled, y, test_size=0.30, random_state=9)


def _get_model(classifier, params):
    default_params = {
        'criterion': 'gini',
        'max_features': None,
        'max_depth': None,
        'min_samples_split': 2,
        'C': 1.0,
        'K': 5
    }

    # Update default parameters with provided params
    params.update(default_params)

    if classifier == 'Decision Tree':
        return DecisionTreeClassifier(criterion=params['criterion'], max_features=params['max_features'],
                                      max_depth=params['max_depth'], min_samples_split=params['min_samples_split'])
    elif classifier == 'SVM':
        return SVC(C=params['C'])
    elif classifier == 'Random Forest':
        return RandomForestClassifier(n_estimators=90)
    elif classifier == 'XGBoost':
        return GradientBoostingClassifier(n_estimators=90)


def _trigger_classifier(classifier, params, X_train, X_test, y_train, y_test):
    if classifier == 'Decision Tree':
        model = _get_model(classifier, params)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        # Calculate ROC, feature importances, and confusion matrix
        feature_importances = model.feature_importances_
        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        f1 = f1_score(y_test, model.predict(X_test))
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        confusion_mat = confusion_matrix(y_test, model.predict(X_test))
        return accuracy, feature_importances, fpr, tpr, thresholds, f1, auc, confusion_mat

    elif classifier == 'SVM':
        model = _get_model(classifier, params)  # Instantiate SVM model
        model.fit(X_train, y_train)  # Fit the model
        accuracy = model.score(X_test, y_test)  # Calculate accuracy
        feature_importances = None
        fpr, tpr, thresholds = roc_curve(y_test, model.decision_function(X_test))
        f1 = f1_score(y_test, model.predict(X_test))
        auc = roc_auc_score(y_test, model.decision_function(X_test))
        confusion_mat = confusion_matrix(y_test, model.predict(X_test))
        return accuracy, feature_importances, fpr, tpr, thresholds, f1, auc, confusion_mat

    elif classifier == 'Random Forest':
        model = _get_model(classifier, params)  # Instantiate Random Forest model
        model.fit(X_train, y_train)  # Fit the model
        accuracy = model.score(X_test, y_test)  # Calculate accuracy
        feature_importances = model.feature_importances_
        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        f1 = f1_score(y_test, model.predict(X_test))
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        confusion_mat = confusion_matrix(y_test, model.predict(X_test))
        return accuracy, feature_importances, fpr, tpr, thresholds, f1, auc, confusion_mat
    
    elif classifier == 'XGBoost':
        model = _get_model(classifier, params)  # Instantiate XGBoost model
        model.fit(X_train, y_train)  # Fit the model
        accuracy = model.score(X_test, y_test)  # Calculate accuracy
        feature_importances = None  # XGBoost doesn't provide feature importances
        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        f1 = f1_score(y_test, model.predict(X_test))
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        confusion_mat = confusion_matrix(y_test, model.predict(X_test))
        return accuracy, feature_importances, fpr, tpr, thresholds, f1, auc, confusion_mat
