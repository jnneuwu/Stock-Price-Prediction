"""
PySpark preprocessing pipeline.

Reads raw OHLCV rows from MySQL (table `stock_daily`), builds 10 features
per row, normalises them to [0, 1] per ticker, marks an 80/20 train/test
split by time, and writes the result to MySQL table `stock_processed`.

Run with:  python pyspark_preprocessing.py
"""
import os
import shutil

# Tell PySpark where Hadoop lives on this Windows machine
os.environ['HADOOP_HOME'] = 'D:\\hadoop'
os.environ['PATH'] = os.environ['PATH'] + ';D:\\hadoop\\bin'

# Same cert-into-ASCII-path workaround used by get_data.py
cert_dir = "D:\\certs"
cert_file = os.path.join(cert_dir, "cacert.pem")
os.makedirs(cert_dir, exist_ok=True)
if not os.path.exists(cert_file):
    import certifi
    shutil.copy(certifi.where(), cert_file)
os.environ['CURL_CA_BUNDLE'] = cert_file
os.environ['SSL_CERT_FILE'] = cert_file

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# Start a Spark session that can talk to MySQL via JDBC
JDBC_JAR = "D:\\mysql-connector-j.jar"
if not os.path.exists(JDBC_JAR):
    print("ERROR: MySQL JDBC driver not found.")
    print(f"Put the .jar file at: {JDBC_JAR}")
    exit()

spark = (
    SparkSession.builder
    .appName("StockDataPreprocessing")
    .master("local[*]")
    .config("spark.driver.extraClassPath", JDBC_JAR)
    .config("spark.jars", JDBC_JAR)
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
print("Spark session ready.")

JDBC_URL = "jdbc:mysql://localhost:3306/stock_prediction"
DB_PROPS = {
    "user": "root",
    "password": os.environ.get("MYSQL_PASSWORD", ""),
    "driver": "com.mysql.cj.jdbc.Driver",
}


# Load raw daily rows from MySQL
print("Reading data from MySQL...")
df = spark.read.jdbc(url=JDBC_URL, table="stock_daily", properties=DB_PROPS)
print(f"Total rows loaded: {df.count()}")
df.show(5)


# Clean: drop nulls, drop duplicates, sort by ticker + date
print("Cleaning data...")
print("Null values per column:")
df.select([F.count(F.when(F.isnull(c), c)).alias(c) for c in df.columns]).show()

df_clean = (
    df.dropna()
      .dropDuplicates(["ticker", "date"])
      .orderBy("ticker", "date")
)
print(f"Rows after cleaning: {df_clean.count()}")


# Feature engineering -- compute 10 technical indicators
print("Calculating technical indicators...")

# Rolling windows per ticker, ordered by date
w7  = Window.partitionBy("ticker").orderBy("date").rowsBetween(-6,  0)
w12 = Window.partitionBy("ticker").orderBy("date").rowsBetween(-11, 0)
w14 = Window.partitionBy("ticker").orderBy("date").rowsBetween(-13, 0)
w20 = Window.partitionBy("ticker").orderBy("date").rowsBetween(-19, 0)
w26 = Window.partitionBy("ticker").orderBy("date").rowsBetween(-25, 0)
w_prev = Window.partitionBy("ticker").orderBy("date")

df_feat = (
    df_clean
    # Simple moving averages (7 and 20 days)
    .withColumn("sma_7",  F.avg("close").over(w7))
    .withColumn("sma_20", F.avg("close").over(w20))
    # Exponential moving averages (12 and 26 days)
    .withColumn("ema_12", F.avg("close").over(w12))
    .withColumn("ema_26", F.avg("close").over(w26))
    # MACD = EMA_12 - EMA_26 (momentum)
    .withColumn("macd", F.col("ema_12") - F.col("ema_26"))
)

# RSI (14-day Relative Strength Index)
df_feat = (
    df_feat
    .withColumn("prev_close", F.lag("close", 1).over(w_prev))
    .withColumn("change", F.col("close") - F.col("prev_close"))
    .withColumn("gain",  F.when(F.col("change") > 0, F.col("change")).otherwise(0))
    .withColumn("loss",  F.when(F.col("change") < 0, F.abs(F.col("change"))).otherwise(0))
    .withColumn("avg_gain", F.avg("gain").over(w14))
    .withColumn("avg_loss", F.avg("loss").over(w14))
    .withColumn("rs", F.col("avg_gain") /
                F.when(F.col("avg_loss") == 0, 0.0001).otherwise(F.col("avg_loss")))
    .withColumn("rsi", 100 - (100 / (1 + F.col("rs"))))
)

# Daily return (%) and 7-day price volatility
df_feat = (
    df_feat
    .withColumn("daily_return",
                (F.col("close") - F.col("prev_close")) / F.col("prev_close") * 100)
    .withColumn("volatility_7", F.stddev("close").over(w7))
)

# Drop the intermediate columns we built RSI from
df_feat = (
    df_feat
    .drop("prev_close", "change", "gain", "loss", "avg_gain", "avg_loss", "rs")
    .dropna()
)


# Drop the first 26 rows per ticker (warm-up period for indicators)
print("Removing the first 26 rows per ticker (unstable indicators)...")
w_row_num = Window.partitionBy("ticker").orderBy("date")
df_feat = (
    df_feat.withColumn("day_num", F.row_number().over(w_row_num))
           .filter(F.col("day_num") > 26)
           .drop("day_num")
)
print(f"Rows after warm-up cut: {df_feat.count()}")


# Remove outliers (daily return outside +/- 50 %)
outliers = df_feat.filter((F.col("daily_return") > 50) | (F.col("daily_return") < -50)).count()
print(f"Outliers found (|daily_return| > 50%): {outliers}")
df_feat = df_feat.filter((F.col("daily_return") <= 50) & (F.col("daily_return") >= -50))
print(f"Rows after outlier cut: {df_feat.count()}")

# Sanity checks
print("\nSample AAPL data after cleaning:")
df_feat.filter(F.col("ticker") == "AAPL") \
    .select("ticker", "date", "close", "sma_7", "sma_20", "macd", "rsi", "daily_return") \
    .show(5)

print("RSI range (should sit roughly inside 0-100):")
df_feat.select(F.min("rsi").alias("min_rsi"),
               F.max("rsi").alias("max_rsi"),
               F.avg("rsi").alias("avg_rsi")).show()

print("Daily-return range:")
df_feat.select(F.min("daily_return").alias("min_return"),
               F.max("daily_return").alias("max_return"),
               F.avg("daily_return").alias("avg_return")).show()


# Normalise every feature to [0, 1] per ticker (MinMax)
print("Applying MinMax normalisation per ticker...")
feature_cols = [
    "open", "high", "low", "close", "adj_close", "volume",
    "sma_7", "sma_20", "ema_12", "ema_26", "macd", "rsi",
    "daily_return", "volatility_7",
]
w_all = Window.partitionBy("ticker")
for col_name in feature_cols:
    min_col = F.min(col_name).over(w_all)
    max_col = F.max(col_name).over(w_all)
    df_feat = df_feat.withColumn(
        f"{col_name}_norm",
        (F.col(col_name) - min_col) / (max_col - min_col),
    )
print("Normalisation done.")


# Train/test split (first 80 % by time = train, last 20 % = test)
print("Splitting train / test...")
w_row = Window.partitionBy("ticker").orderBy("date")
df_feat = (
    df_feat
    .withColumn("row_num",     F.row_number().over(w_row))
    .withColumn("total_count", F.count("*").over(w_all))
    .withColumn("split",
                F.when(F.col("row_num") <= F.col("total_count") * 0.8, "train")
                 .otherwise("test"))
)
print(f"Training rows: {df_feat.filter(F.col('split') == 'train').count()}")
print(f"Test     rows: {df_feat.filter(F.col('split') == 'test').count()}")


# Save processed rows back to MySQL (overwrite table `stock_processed`)
print("Saving processed data to MySQL...")
df_feat = df_feat.drop("row_num", "total_count")
df_feat.write.jdbc(
    url=JDBC_URL,
    table="stock_processed",
    mode="overwrite",
    properties=DB_PROPS,
)
print("Saved to stock_processed table.")


# Final verification: read back and show per-ticker split counts
print("\nFinal Verification")
df_check = spark.read.jdbc(url=JDBC_URL, table="stock_processed", properties=DB_PROPS)
print(f"Total rows in stock_processed: {df_check.count()}")
print(f"Total columns: {len(df_check.columns)}")
df_check.groupBy("ticker", "split").count().orderBy("ticker", "split").show(20)

spark.stop()
print("\nDone. PySpark preprocessing complete.")
