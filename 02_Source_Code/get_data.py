"""
Download 5 years of daily OHLCV data for 10 tickers from Yahoo Finance
and save into the MySQL table `stock_daily`.

Re-running is safe -- ON DUPLICATE KEY UPDATE refreshes existing rows.

Run with:  python get_data.py
"""
import os
import shutil
from datetime import datetime

# Copy the certifi cert into an ASCII path. yfinance refuses to follow a path
# with non-ASCII characters on Windows, so we keep the file at D:\certs\.
cert_dir = "D:\\certs"
cert_file = os.path.join(cert_dir, "cacert.pem")
os.makedirs(cert_dir, exist_ok=True)
if not os.path.exists(cert_file):
    import certifi
    shutil.copy(certifi.where(), cert_file)
    print(f"Copied certificate to {cert_file}")

os.environ['CURL_CA_BUNDLE'] = cert_file
os.environ['SSL_CERT_FILE'] = cert_file
os.environ['REQUESTS_CA_BUNDLE'] = cert_file

import pandas as pd
import yfinance as yf
import mysql.connector

# 10 tickers: 8 US large-caps + 1 Korean + 1 Taiwanese
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSM", "AVGO", "005930.KS", "TSLA",
]

# Pull 5 years of daily data up to today
START_DATE = "2020-03-04"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password=os.environ.get("MYSQL_PASSWORD", ""),
    database="stock_prediction",
)
cursor = conn.cursor()

# Upsert SQL -- insert new rows or refresh the values on existing ones
INSERT_SQL = """
    INSERT INTO stock_daily (ticker, date, open, high, low, close, adj_close, volume)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        open = VALUES(open),
        high = VALUES(high),
        low = VALUES(low),
        close = VALUES(close),
        adj_close = VALUES(adj_close),
        volume = VALUES(volume)
"""

# Download one ticker at a time and insert into MySQL
for ticker in TICKERS:
    print(f"Downloading {ticker}...")
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if df.empty:
            print(f"  WARNING: no data for {ticker}, skipping.")
            continue

        df.reset_index(inplace=True)

        # yfinance sometimes returns columns as a MultiIndex like ('Close','AAPL').
        # Flatten that back to a normal single-level column name.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if c[1] in ('', ticker) else c[0] for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]

        count = 0
        for _, row in df.iterrows():
            adj_close = row["Adj Close"] if "Adj Close" in df.columns else row["Close"]
            cursor.execute(INSERT_SQL, (
                ticker,
                row["Date"].strftime("%Y-%m-%d"),
                round(float(row["Open"]), 4),
                round(float(row["High"]), 4),
                round(float(row["Low"]), 4),
                round(float(row["Close"]), 4),
                round(float(adj_close), 4),
                int(row["Volume"]),
            ))
            count += 1

        conn.commit()
        print(f"  Done. Inserted {count} rows for {ticker}")
    except Exception as e:
        print(f"  ERROR with {ticker}: {e}")
        conn.rollback()

# Quick check: how many rows per ticker did we end up with?
print("\n========== Data Verification ==========")
cursor.execute("""
    SELECT ticker, COUNT(*) AS rows_count,
           MIN(date) AS start_date, MAX(date) AS end_date
      FROM stock_daily
     GROUP BY ticker
     ORDER BY ticker
""")
total = 0
for ticker, rows_count, start, end in cursor.fetchall():
    print(f"  {ticker}: {rows_count} rows ({start} to {end})")
    total += rows_count
print(f"\n  Total rows in database: {total}")

cursor.close()
conn.close()
print("\nDone. All data saved to MySQL.")
