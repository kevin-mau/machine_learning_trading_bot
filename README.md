# Machine Learning Trading Bot

## Overview

Machine trading is advantageous because of its speed but people still need to specifically program these systems, which limits their ability to adapt
to new data. In this project, we will attempt to improve an existing algorithmic trading systems. To do so, we’ll enhance the existing trading signals
with machine learning algorithms that can adapt to new data and evolving markets.

In this Jupyter notebook, we’ll do the following:

Implement an algorithmic trading strategy that uses machine learning to automate the trade decisions.

Adjust the input parameters to optimize the trading algorithm.

Train a new machine learning model and compare its performance to that of a baseline model.

We will also create a report that compares the performance of the machine learning models based on the trading predictions that each makes and the
resulting cumulative strategy returns.

## Process

We will use a dataset of historical lending activity from a peer-to-peer lending services company to build these models.  The goal of building 
these models is so that we can identify the creditworthiness of borrowers.  The creditworthiness will be signified by a value of `0` in the 
`loan_status` column, meaning that the loan is healthy.  A value of `1` means that the loan has a high risk of defaulting.

To build the model, first we take the CSV dataset and put it into a dataframe.  We will split the dataframe with `y` being the `loan_status` column, 
and `X` dataframe as the remaining columns.  Here we use the `value_counts` function to show us the amount of healthy loans in the dataset versus
the amount of risky loans in both the original model.  Finally, we split the data into training and testing datasets by using the function
`train_test_split`.  With the training datasets we can fit our logistic regression model.  We can predict and evaluate the model’s performance by
calculating the accuracy score of the model, generating a confusion matrix, and printing the classification report.

From the `value_counts` function we did earlier, we see that the 75036 number of good loans greatly outweight the 2500 risky loans, therefore we 
predict a new logistic regression model with resampling the training data by oversampling the high-risk loans.  We use the `RandomOverSampler`
module from the imbalanced-learn library to resample the data and with `value_counts` function we confirm that the labels have an equal number of data 
points.  Once again, we use the `LogisticRegression` classifier, but this time on the oversampled data and print and generate the same reports that we 
did from the first test in order to directly compare how our oversampled test model did.  In the next section, we will examine these results.

## Results

* Machine Learning Model 1:
  * The balanced accuracy score of Model 1 is 0.952
  * The precision score of the `0` class is 1.0 and `1` class is 0.85.
  * The recall scores of the `0` class is 0.99 and `1` class is 0.91.
![classification_report_1](https://github.com/kevin-mau/credit_risk_resampling/blob/main/Resources/classification_report_1.PNG?raw=true)

* Machine Learning Model 2 (Oversampled model):
  * The balanced accuracy score of Model 1 is 0.993
  * The precision score of the `0` class is 1.0 and `1` class is 0.84.
  * The recall scores of the `0` class is 0.99 and `1` class is 0.99.
![classification_report_2](https://github.com/kevin-mau/credit_risk_resampling/blob/main/Resources/classification_report_2.PNG?raw=true)


## Summary

According to the balanced accuracy score, model 2 would be the better model.  The performance is not the only factor to weigh on as it is more important
for us to predict the value of `1`, the high-risk loan.  In random oversampling, we randomly select instances of the risky loans class and add them to the
training set until we’ve balanced the majority (healthy loans) and minority classes (high-risk loans).  Since it artificially increases the number of instances
in the minority class, it increases the frequency of the `1` values.  That trains the model to tend to correctly predict all the true `1` values (to have much
higher recall at 0.99 vs 0.91). But, this happens at the expense of tending to overestimate the frequency of the `1` values (to have a lower precision at 0.84
vs 0.85).  Since our model is dealing with risky defaulting loans, the trade-off is justified as it is more important to predict the `1`'s in this case.  In
summary, as we are trying to build a model that predicts dealing with potential loan defaults, it would be more prudent to go with the oversampled model.

---

## Data:

The "emerging_markets_ohlcv.csv" file is a CSV file of historical OHLCV market data.  OHLCV is an aggregated form of market data standing for Open, High, Low, Close and Volume.

---

## Contributors

kevin-mau

---

## License

MIT
