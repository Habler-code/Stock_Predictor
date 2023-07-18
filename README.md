# Stock Predictor

A Simple stock price prediction program using Long Short-Term Memory (LSTM), Keras (Tensorflow).

The program fetches historical stock price data using the Yfinance Lin (Yahoo finance).

Trains an LSTM model on the data, and then uses the model to predict future stock prices.


# Long Short-Term Memory (LSTM) #
LSTM is an artificial recurrent neural network (RNN) architecture used in the field of deep learning.

LSTM networks are well-suited to classifying, processing and making predictions based on time series data, since there can be lags of unknown duration between important events in a time series.

For more information about LSTMs: [LSTM](https://medium.com/@kangeugine/long-short-term-memory-lstm-concept-cb3283934359). 


<p align="center">
<img src="https://miro.medium.com/v2/resize:fit:388/1*hG4zBCCRq18oi8aarj-owA.png" width="300" height="300">
</p>


**Disclaimer:**
This script was built only for educational purposes. Do not use it for trading.

Risk under your control.

**Required Libraries:**
```
pip install -r requirements.txt
```
from venv:
```
# Create a virtual environment
python3 -m venv <venvname>
# Activate the virtual environment
.\<venvname>\Scripts\activate
pip install -r requirements.txt
```
**Usage**

1.Clone this repository or download the script file.

2.Install the dependencies mentioned in the prerequisites section.

3.Run the script using the following command:

```
python ./lstm_stock_predictor.py
```

When prompted, enter the stock name you want to predict:
```
Enter the ticker symbol of the stock:: <Stock Name (e.g., AMZN, GOOGL)>
```
