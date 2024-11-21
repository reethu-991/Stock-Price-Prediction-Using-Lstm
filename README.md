                             STOCK PRICE PREDICTION USING LSTM 
A vital component of financial research is stock price prediction, This helps investors and traders 
make informed decisions about buying or selling stocks.This study proposes a model for predicting 
stock prices that predicts the closing price of Google stocks by using Recurrent Neural Networks  
and Long Short-Term Memory. The Google stock dataset is used to train the suggested model at 
first. The Yahoo Finance API was used to gather historical stock data over a 20-year span. To give 
trend insights, important parameters including percentage changes, adjusted closing prices, and 
moving averages for 100 and 250 days were calculated. Sequences of 100 days were utilized as 
input to forecast the following day's stock price following the dataset had been preprocessed by 
standardizing the values using MinMaxScaler. 
The LSTM-based model, including two LSTM layers succeeded by dense layers was assessed 
using 30% of the dataset after being trained on 70% of it. The model demonstrated strong 
performance with a Root Mean Square Error (RMSE) of roughly 4.31, signifying its precision in 
predicting stock prices. The predictions were then inversely transformed to match the original scale 
of stock prices for validation. 
To improve the prediction accuracy, we also employed a range of machine learning (ML) models, 
including Random Forest, Decision Tree, XGBoost, and Voting Classifier. 
Furthermore, Streamlit was used to deploy the model in an intuitive interface that let users enter 
stock IDs and view projected prices in addition to moving averages. This study shows how well 
LSTM handles time-series data and how it may be used for real-time stock market forecasting and 
decision-making.
