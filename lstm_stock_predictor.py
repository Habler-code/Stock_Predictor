import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from sklearn.metrics import mean_squared_error, r2_score

class StockPredictor:
    def __init__(self, ticker_symbol):
        self.ticker_symbol = ticker_symbol
        self.ticker_data = yf.Ticker(ticker_symbol)
        self.scaler = MinMaxScaler(feature_range = (0, 1))

    def fetch_data(self, start_date, end_date):
        return self.ticker_data.history(period='1d', start=start_date, end=end_date)

    def preprocess_data(self, data, steps=60):
        training_set = data.iloc[:, 1:2].values
        training_set_scaled = self.scaler.fit_transform(training_set)

        X_train = []
        y_train = []

        for i in range(steps, len(training_set)):
            X_train.append(training_set_scaled[i-steps:i, 0])
            y_train.append(training_set_scaled[i, 0])

        X_train = np.array(X_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        y_train = np.array(y_train)

        return X_train, y_train

    def build_model(self, X_train):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        return model


    def train_model(self, model, X_train, y_train, epochs=100, batch_size=32):
        model.compile(optimizer='adam', loss='mean_squared_error')
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        return history

    def predict_stock_price(self, model, data, future_days):
        dataset_total = pd.concat((self.dataset_train['Open'], data['Open']), axis = 0)
        inputs = dataset_total[len(dataset_total) - len(data) - future_days:].values

        inputs = inputs.reshape(-1,1)
        inputs = self.scaler.transform(inputs)

        X_test = []
        inputs_flattened = inputs.flatten()

        for i in range(future_days, len(inputs)):
            X_test.append(inputs_flattened[i-future_days:i])

        X_test = np.array(X_test, dtype=object)
        X_test = np.vstack(X_test).astype(float)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted_stock_price = []

        for i in range(future_days):
            prediction = model.predict(np.array([X_test[i, :]]))
            predicted_stock_price.append(prediction)
            
            # Add the new prediction to X_test for use in next prediction
            X_test = np.append(X_test, prediction)[1:]
            X_test = np.reshape(X_test, (1, X_test.shape[0], 1))

        predicted_stock_price = np.array(predicted_stock_price)
        predicted_stock_price = predicted_stock_price.reshape(predicted_stock_price.shape[0], -1)
        predicted_stock_price = self.scaler.inverse_transform(predicted_stock_price)

        return predicted_stock_price


def main():
    ticker_symbol = input("Enter the ticker symbol of the stock: ")
    predictor = StockPredictor(ticker_symbol)

    start_date = '2010-1-1'
    end_date = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')

    predictor.dataset_train = predictor.fetch_data(start_date, end_date)
    X_train, y_train = predictor.preprocess_data(predictor.dataset_train)

    model = predictor.build_model(X_train)
    history = predictor.train_model(model, X_train, y_train)

    future_days = 20
    dataset_test = predictor.fetch_data(end_date, (datetime.today() + timedelta(days=future_days)).strftime('%Y-%m-%d'))
    predicted_stock_price = predictor.predict_stock_price(model, dataset_test, future_days)

    y_train = y_train.flatten()
    predicted_stock_price = predicted_stock_price.flatten()

    mse = mean_squared_error(y_train[-len(predicted_stock_price):], predicted_stock_price)
    r2 = r2_score(y_train[-len(predicted_stock_price):], predicted_stock_price)
    print(f"MSE: {mse}, R2 Score: {r2}")

    print("Future prices for the next 20 days:")
    for i in range(1, future_days+1):
        print(f"Day {i}: {predicted_stock_price[-i]}")

if __name__ == "__main__":
    main()
