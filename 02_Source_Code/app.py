"""
Flask backend for the Stock Price Prediction dashboard.

Six endpoints. Five of them read from MySQL, one reads a JSON file.
No model is loaded here -- train_1.py is the only writer.
Run with:  python app.py    then open http://localhost:5000
"""
import os
import json
import mysql.connector
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# MySQL connection details
DB = dict(
    host="localhost",
    user="root",
    password=os.environ.get("MYSQL_PASSWORD", ""),
    database="stock_prediction",
)


def query_db(sql, params=(), as_dict=True):
    """Run one SELECT and return rows. Closes the connection for you."""
    conn = mysql.connector.connect(**DB)
    cursor = conn.cursor(dictionary=as_dict)
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


# Serve the front-end page (templates/index.html)
@app.route('/')
def index():
    return render_template('index.html')


# List of the 10 ticker codes (for the dropdown in the UI)
@app.route('/api/tickers')
def get_tickers():
    rows = query_db(
        "SELECT DISTINCT ticker FROM stock_processed ORDER BY ticker",
        as_dict=False,
    )
    return jsonify([r[0] for r in rows])


# Daily OHLCV plus the 9 indicator features for one ticker
@app.route('/api/stock/<ticker>')
def get_stock(ticker):
    rows = query_db(
        """SELECT date, open, high, low, close, volume,
                  sma_7, sma_20, ema_12, ema_26, macd, rsi,
                  daily_return, volatility_7, split
             FROM stock_processed
            WHERE ticker = %s
            ORDER BY date""",
        (ticker,),
    )
    for r in rows:
        r['date'] = str(r['date'])
        for k, v in r.items():
            if isinstance(v, float):
                r[k] = round(v, 4)
    return jsonify(rows)


# Held-out predictions: actual close vs each of the 5 model outputs
@app.route('/api/predictions/<ticker>')
def get_predictions(ticker):
    rows = query_db(
        """SELECT date, actual, lstm_pred, rnn_pred, gru_pred, cnn_pred, rf_pred
             FROM model_predictions
            WHERE ticker = %s
            ORDER BY date""",
        (ticker,),
    )
    for r in rows:
        r['date'] = str(r['date'])
    return jsonify(rows)


# Per-stock RMSE / MAE table (read from JSON cache for speed)
@app.route('/api/model_results')
def get_model_results():
    with open('model_results.json', 'r') as f:
        return jsonify(json.load(f))


# 5-day recursive forecast for one ticker
@app.route('/api/forecast/<ticker>')
def get_forecast(ticker):
    rows = query_db(
        """SELECT day_ahead, forecast_date, actual,
                  lstm_pred, rnn_pred, gru_pred, cnn_pred, rf_pred
             FROM model_forecasts
            WHERE ticker = %s
            ORDER BY day_ahead""",
        (ticker,),
    )
    for r in rows:
        r['forecast_date'] = str(r['forecast_date'])
        for k in ('actual', 'lstm_pred', 'rnn_pred', 'gru_pred', 'cnn_pred', 'rf_pred'):
            if r[k] is not None:
                r[k] = round(float(r[k]), 8)
    return jsonify(rows)


# Training-loss history for the loss-curve chart on the dashboard
@app.route('/api/loss_history')
def get_loss_history():
    try:
        with open('loss_history.json', 'r') as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
