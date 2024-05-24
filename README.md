# Boston House Price Prediction

This project aims to predict the prices of houses in Boston based on various features using machine learning models. We utilize the Boston Housing dataset from the UCI Machine Learning Repository to train and evaluate our models.

## Dataset

The [Boston Housing dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names) comprises 506 samples, with each sample representing a town or suburb in Boston. The dataset contains 13 attributes (features) along with a target column, which is the median value of owner-occupied homes in thousands of dollars. Here are some key features present in the dataset:

- **CRIM**: Per capita crime rate by town
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq.ft.
- **INDUS**: Proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX**: Nitric oxides concentration (parts per 10 million)
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built prior to 1940
- **DIS**: Weighted distances to five Boston employment centers
- **RAD**: Index of accessibility to radial highways
- **TAX**: Full-value property-tax rate per $10,000
- **PTRATIO**: Pupil-teacher ratio by town
- **B**: 1000(Bk - 0.63)^2 where Bk is the proportion of Blacks by town
- **LSTAT**: Percentage of lower status of the population

## Project Overview

### Data Exploration and Preprocessing

We begin by loading the dataset and performing exploratory data analysis (EDA) to understand its structure and characteristics. This involves checking for missing values, visualizing distributions, and exploring correlations between features using techniques like heatmaps.

### Model Building

We train several machine learning models on the dataset, including:

####Linear Regression<br>
  ![Histogram](https://github.com/mihul04saxena/House-Price-Prediction/blob/main/images/LR-Histogram.png)
  ![Visualization](https://github.com/mihul04saxena/House-Price-Prediction/blob/main/images/LR-Visualization.png)
  ![Check residual](https://github.com/mihul04saxena/House-Price-Prediction/blob/main/images/LR-Check_residual.png)
####Random Forest Regressor<br>
  ![Visualization](https://github.com/mihul04saxena/House-Price-Prediction/blob/main/images/RF-Visualization.png)
  ![Check residual](https://github.com/mihul04saxena/House-Price-Prediction/blob/main/images/RF-check_residual.png)
####XGBoost Regressor<br>
  ![Visualization](https://github.com/mihul04saxena/House-Price-Prediction/blob/main/images/SVM-visualization.png)
  ![Check residual](https://github.com/mihul04saxena/House-Price-Prediction/blob/main/images/SVM-check_residual.png)
####Support Vector Machines (SVM) Regressor<br>
  ![Visualization](https://github.com/mihul04saxena/House-Price-Prediction/blob/main/images/XGBR-Visualization.png)
  ![Check residual](https://github.com/mihul04saxena/House-Price-Prediction/blob/main/images/XGBR-check_residual.png)

For each model, we split the dataset into training and testing sets, train the model on the training data, and then evaluate its performance on the testing data using metrics like R-squared score, mean absolute error (MAE), mean squared error (MSE), and root mean squared error (RMSE).

### Model Evaluation and Comparison

After training and evaluating each model, we compare their performance metrics to determine which model performs the best for predicting house prices in Boston.

## Conclusion

Based on the evaluation results, we identify the XGBoost Regressor as the best-performing model for this dataset, achieving the highest R-squared score among all the models tested.
