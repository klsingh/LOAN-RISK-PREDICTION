# LOAN RISK PREDICTION

## Required Libraries
pandas: A library used for data manipulation and analysis.
numpy: A library used for scientific computing.
sklearn.linear_model: A module used for implementing various linear models, such as Logistic Regression.
sklearn.model_selection: A module used for splitting the dataset into training and testing sets.
sklearn.metrics: A module used for evaluating the model's performance.

## Loading Data
To load the dataset, you can use the read_csv() method of pandas library. In this code, the dataset is loaded from a CSV file named `lending_club_data.csv`.

## Preprocessing Data
Preprocessing the data is an important step before building a prediction model. In this code, we will perform the following preprocessing steps:
Drop columns with more than 50% missing values
Drop irrelevant columns
Drop columns related to future or post loan
Encode categorical variables
Fill missing values with median

## Creating Feature Matrix and Target Variable
The next step is to define the feature matrix and target variable. In this code, we will define the target variable as a binary variable that takes the value 1 if the loan is fully paid and 0 otherwise. 
We will also define the feature matrix that consists of all the remaining columns in the dataset.

## Splitting Data into Training and Testing Sets
To evaluate the model's performance, we need to split the dataset into training and testing sets. In this code, we will split the dataset into 80% training data and 20% testing data using the `train_test_split()` method from the `sklearn.model_selection` module.

## Training the Model
The next step is to train the model using Logistic Regression. In this code, we will use the `LogisticRegression()` method from the `sklearn.linear` model module to train the model.

## Evaluating Model Performance
To evaluate the model's performance, we will calculate the accuracy score on the testing data. In this code, we will use the `accuracy_score(`) method from the `sklearn.metrics` module.

## Using the Prediction Model for Investment Decisions
Investors can use prediction models to make informed decisions about their investment portfolio. In this code, the prediction model can be used to predict whether a loan will be fully repaid or not. Based on this prediction, investors can decide whether to invest in a particular loan or not. If the model predicts that the loan will be fully repaid, then the investor can consider investing in that loan.

model predicts that the loan will be fully repaid or not. If the model predicts that the loan is likely to be fully repaid, the investor may feel more confident in investing in that loan. On the other hand, if the model predicts that the loan is not likely to be fully repaid, the investor may choose not to invest in that loan or to invest with caution.

Investors can also use prediction models to help diversify their investment portfolio. By using models like this one to predict the likelihood of loan repayment, investors can make more informed decisions about which loans to invest in and how to balance their investment portfolio.

Overall, prediction models can be a useful tool for investors in making investment decisions, but it is important to keep in mind that no model is perfect and there is always some level of risk involved in investing. 
