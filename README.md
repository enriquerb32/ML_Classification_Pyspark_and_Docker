![](https://assets.website-files.com/5dc3b47ddc6c0c2a1af74ad0/5e0a328bedb754beb8a973f9_logomark_website.png)

# Streamlit Healthcare Machine Learning Data App

![](extra/StreamlitML.gif)

## Objective
The aim of this project is to build a PySpark ML model to analyze medical data for predictive purposes. Therefore, the data preprocessing includes transformations of its variables to improve the explainability of the entire model. Despite our main focus on PySpark, most of the calculations and plots are replicated using Scikit Learn ML packages. This approach allows us to identify differences in the statistical results by simply changing the Python packages. PySpark modules utilize Decision Tree and Random Forest algorithms, while Scikit Learn incorporates these two algorithms as well as SVM and XGBoost.

## Key Features
* Interactive Interface: Explore and visualize the dataset's features, correlations, and distributions through an intuitive Streamlit interface.
* Correlation Matrix: Visualize the correlation matrix to identify relationships between different features in the dataset.
* Histograms and Plots: Gain insights into key features such as age, Body Mass Index (BMI), and Mean Arterial Pressure (MAP) through interactive histograms and scatterplots.
* Classifier Selection: Choose from a variety of classifiers via the sidebar and customize parameters to train and evaluate machine learning models.
* Model Evaluation: Evaluate the performance of each machine learning model with metrics such as accuracy, F1 score, AUC, and confusion matrix.
* Predictions: Obtain predictions on cardiovascular disease risk based on age using PySpark.
* Provide Tuning parameters in the UI 

## Testing and Continuous Integration

* Comprehensive Testing: Utilize pytest for unit tests, flake8 for code linting, isort for import sorting, black for code formatting, pylint for static analysis, and bandit for security checks.
* Dockerized Environment: Docker is used for containerization, ensuring consistency and reproducibility across different environments.
* Continuous Integration: Set up continuous integration to maintain code quality and reliability through automated testing and validation.

## Running the App

1. Clone the repository
 ```buildoutcfg
git clone https://github.com/enriquerb32/streamlit-ML-Classification-webapp.git
cd streamlit-healthcare-ml-app
``` 
2. Build and run the docker image
```buildoutcfg
docker build --tag streamlit-healthcare:1.0 .
docker run -it -p 8501:8501 streamlit-healthcare:1.0
```
3. Browse the [url](http://localhost:8501)

## Contributions and Future Enhancements

Contributions to this project are welcome! Feel free to fork the repository and submit pull requests for improvements or additional features. Potential enhancements include:

* Adding support for more machine learning algorithms
* Enhancing the user interface with additional visualizations and interactivity
* Integrating advanced model evaluation techniques and hyperparameter tuning, either UI-based or optimising it on the code

## License

This project is licensed under the MIT License
