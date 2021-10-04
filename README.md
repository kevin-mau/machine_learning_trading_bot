# Machine Learning Trading Bot

## Overview

Machine trading is advantageous because of its speed but people still need to specifically program these systems which limits their ability to adapt
to new data. In this project we will attempt to improve an existing algorithmic trading systems. To do so we’ll enhance the existing trading signals
with machine learning algorithms that can adapt to new data and evolving markets.

In this Jupyter notebook, we’ll do the following:

  * Implement an algorithmic trading strategy that uses machine learning to automate the trade decisions.

  * Adjust the input parameters to optimize the trading algorithm.

  * Train a new machine learning model and compare its performance to that of a baseline model.

We will also create a report that compares the performance of the machine learning models based on the trading predictions that each makes and the
resulting cumulative strategy returns.

## Process

We will use a historical OHLCV market dataset in a CSV file and read into a dataframe.  Now that it is in the dataframe, we will locate the `close` 
column create a new `Actual Returns` column which is the percent change of the `close` column.  With just these columns in a new dataframe, we can
add signals of the rolling average.  We will generate the trading signals using short- and long-window SMA values with the baseline model set at 4 as 
the short window and 100 as the long window.  We create the new Signal column with the code below:
```python
    # Initialize the new Signal column
    signals_df['Signal'] = 0.0

    # When Actual Returns are greater than or equal to 0, generate signal to buy stock long
    signals_df.loc[(signals_df['Actual Returns'] >= 0), 'Signal'] = 1

    # When Actual Returns are less than 0, generate signal to sell stock short
    signals_df.loc[(signals_df['Actual Returns'] < 0), 'Signal'] = -1
```
Next, we will split the data into training and testing datasets with `X` as the `SMA_Fast` and the `SMA_Slow` columns and `y` as the `Signal` column.
With the training and testing datasets ready, we can first try training the baseline model.

Our baseline model will use the `SVC` classifier model from SKLearn's support vector machine (SVM) learning method to fit the training data and make
predictions based on the testing data.
```python
    # From SVM, instantiate SVC classifier model instance
    svm_model = svm.SVC()
 
    # Fit the model to the data using the training data
    svm_model = svm_model.fit(X_train_scaled, y_train)
 
    # Use the testing data to make the model predictions
    svm_pred = svm_model.predict(X_test_scaled)
```
In the next section, I will display the baseline model's classification report and a cumulative return plot that shows the actual returns vs. the strategy
returns.  But before that I will discuss how we will modify the windows to try to optimize the trading algorithm.  For this trial, I will adjust to a 
shorter windows strategy, with the using short- and long-window SMA values set at 1 as the short window and 5 as the long window.  

We will also use a second machine learning model.  To try and get more accuracy from our trading algorithm.  On the second ML model, we will use 
`LogisticRegression`.  We backtest the new model with the same training and testing datasets to evaluate its performance. 

## Results

* Baseline SVC Model Classification Report:

![baseline_svm_classification_report](https://github.com/kevin-mau/machine_learning_trading_bot/blob/main/Resources/baseline_svm_classification_report.jpg?raw=true)

* Shorter window SVC Model Classification Report:

![shorter_svm_classification_report](https://github.com/kevin-mau/machine_learning_trading_bot/blob/main/Resources/shorter_svm_classification_report.jpg?raw=true)

* Baseline LR Model Classification Report:

![baseline_lr_classification_report](https://github.com/kevin-mau/machine_learning_trading_bot/blob/main/Resources/baseline_lr_classification_report.jpg?raw=true)

* Shorter window LR Model Classification Report:

![shorter_lr_classification_report](https://github.com/kevin-mau/machine_learning_trading_bot/blob/main/Resources/shorter_lr_classification_report.jpg?raw=true)

* Baseline SVC Model Cumulative Return Plot:

![baseline_svm_return_plot](https://github.com/kevin-mau/machine_learning_trading_bot/blob/main/Resources/baseline_svm_return_plot.jpg?raw=true)

* Shorter window SVC Model Cumulative Return Plot:

![shorter_svm_return_plot](https://github.com/kevin-mau/machine_learning_trading_bot/blob/main/Resources/shorter_svm_return_plot.jpg?raw=true)

* Baseline LR Model Cumulative Return Plot:

![baseline_lr_return_plot](https://github.com/kevin-mau/machine_learning_trading_bot/blob/main/Resources/baseline_lr_return_plot.jpg?raw=true)

* Shorter window LR Model Cumulative Return Plot:

![shorter_lr_return_plot](https://github.com/kevin-mau/machine_learning_trading_bot/blob/main/Resources/shorter_lr_return_plot.jpg?raw=true)



## Summary

As the reports show, we can gain a slight 0.01 in accuracy when using a shorter window.  When using the same training and testing datasets, we were able to get more
accurate results with the SVC classifier model over the LR classification model.

---

## Data:

The "emerging_markets_ohlcv.csv" file is a CSV file of historical OHLCV market data.  OHLCV is an aggregated form of market data standing for Open, High, Low, Close and Volume.

---

## Contributors

kevin-mau

---

## License

MIT
