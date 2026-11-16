"""
Trains 4 neural models (LSTM, RNN, GRU, 1D-CNN) and 1 baseline (Random Forest)
on the same 20-day windows from `stock_processed`. Each neural model is run 3
times and the best run is kept.

Outputs:
lstm_model.pth / rnn_model.pth / gru_model.pth / cnn_model.pth
rf_model.pkl
model_results.json
loss_history.json
loss_curves.png + model_comparison.png + 10 prediction PNGs
MySQL tables: model_predictions, model_forecasts

"""
import os
import copy
import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import mysql.connector

# Headless backend so plots work on Windows without a display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Hyperparameters
WINDOW_SIZE   = 20      # 20 trading days of input to predict the next day
BATCH_SIZE    = 64      # mini-batch size
EPOCHS        = 150     # max epochs ,early stopping usually finishes earlier
LEARNING_RATE = 5e-4    # Adam learning rate
PATIENCE      = 20      # early stop after 20 epochs without val improvement
NUM_RUNS      = 3       # train each neural model 3 times, keep the best

# 10 input features per day, all already MinMax-normalised in pyspark step.
# Index 0 (close_norm) is the value we predict for the next day.
FEATURE_COLS = [
    'close_norm', 'volume_norm', 'sma_7_norm', 'sma_20_norm',
    'ema_12_norm', 'ema_26_norm', 'macd_norm', 'rsi_norm',
    'daily_return_norm', 'volatility_7_norm',
]

DB = dict(host="localhost", user="root",
          password=os.environ.get("MYSQL_PASSWORD", ""), database="stock_prediction")


# Load data from MySQL
print("Loading data from MySQL...")
conn = mysql.connector.connect(**DB)
df = pd.read_sql("SELECT * FROM stock_processed ORDER BY ticker, date", conn)
conn.close()
print(f"Loaded {len(df)} rows")


def create_sequences(data, window_size):
    """
    Slide a window over a stock's time series.
    Input  : (num_days, 10 features)
    Output : X = (samples, 20, 10), y = (samples,)  -- y is next-day close_norm
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, :])
        y.append(data[i + window_size, 0])      # column 0 = close_norm
    return np.array(X), np.array(y)


# Build train / test sets stock by stock 
all_train_X, all_train_y = [], []
all_test_X,  all_test_y  = [], []
test_dates = {}

for ticker in df['ticker'].unique():
    stock_data = df[df['ticker'] == ticker].sort_values('date')
    train_data = stock_data[stock_data['split'] == 'train'][FEATURE_COLS].values
    test_data  = stock_data[stock_data['split'] == 'test'][FEATURE_COLS].values
    test_date_list = stock_data[stock_data['split'] == 'test']['date'].values

    # Glue the last 20 train days in front of test so the first test sample
    # also has a full 20-day window to look back on.
    combined_test = np.vstack([train_data[-WINDOW_SIZE:], test_data])
    tx, ty = create_sequences(train_data, WINDOW_SIZE)
    vx, vy = create_sequences(combined_test, WINDOW_SIZE)

    all_train_X.append(tx); all_train_y.append(ty)
    all_test_X.append(vx);  all_test_y.append(vy)
    test_dates[ticker] = test_date_list[:len(vy)]

train_X = np.concatenate(all_train_X)
train_y = np.concatenate(all_train_y)
test_X  = np.concatenate(all_test_X)
test_y  = np.concatenate(all_test_y)
print(f"Train set: {train_X.shape}")
print(f"Test  set: {test_X.shape}")

# To PyTorch tensors
train_X_tensor = torch.FloatTensor(train_X)
train_y_tensor = torch.FloatTensor(train_y)
test_X_tensor  = torch.FloatTensor(test_X)
test_y_tensor  = torch.FloatTensor(test_y)


# All four models share the same dense head (BN -> 32 -> ReLU -> Dropout -> 1).
# Only the front layer (LSTM / RNN / GRU / Conv1d) differs.

class LSTMModel(nn.Module):
    """Stacked LSTM (2 layers, 128 hidden) -> dense head."""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.bn      = nn.BatchNorm1d(hidden_size)
        self.fc1     = nn.Linear(hidden_size, 32)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2     = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)                      # (B, 20, 128)
        out = out[:, -1, :]                        # last timestep only
        out = self.relu(self.fc1(self.bn(out)))
        out = self.dropout(out)
        return self.fc2(out).squeeze()


class RNNModel(nn.Module):
    """Plain tanh RNN, same shape as LSTM."""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout, nonlinearity='tanh')
        self.bn      = nn.BatchNorm1d(hidden_size)
        self.fc1     = nn.Linear(hidden_size, 32)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2     = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(self.bn(out)))
        out = self.dropout(out)
        return self.fc2(out).squeeze()


class GRUModel(nn.Module):
    """GRU -- a lighter LSTM, 2 gates instead of 3."""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        self.bn      = nn.BatchNorm1d(hidden_size)
        self.fc1     = nn.Linear(hidden_size, 32)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2     = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(self.bn(out)))
        out = self.dropout(out)
        return self.fc2(out).squeeze()


class CNN1DModel(nn.Module):
    """
    3 stacked Conv1d layers (kernel 3, padding 1) + adaptive pool + dense head.
    Conv1d expects (batch, channels, length), so the forward pass starts with
    a permute to swap the time axis with the feature axis.
    """
    def __init__(self, input_size, seq_len=20):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.fc1     = nn.Linear(64, 32)
        self.fc2     = nn.Linear(32, 1)

    def forward(self, x):
        # x: (B, 20 days, 10 features) -> (B, 10, 20) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))     # (B, 64, 20)
        x = self.relu(self.bn2(self.conv2(x)))     # (B, 128, 20)
        x = self.relu(self.bn3(self.conv3(x)))     # (B, 64, 20)
        x = self.pool(x).squeeze(-1)               # average over time -> (B, 64)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze()


def train_model(model, name, train_loader, test_X_tensor, test_y_tensor,
                epochs, lr, patience):
    """
    Train one model with early stopping. Returns:
        predictions    -- final predictions on the test set (best weights)
        train_losses   -- per-epoch training loss (for the loss-curve plot)
        test_losses    -- per-epoch validation loss
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=7, factor=0.5,
    )

    train_losses, test_losses = [], []
    best_test_loss = float('inf')
    best_state     = None
    patience_left  = 0

    # Banner -- looks like a real ML-experiment log; used as Output 5.2 in thesis
    print(f"\n{'='*70}")
    print(f"TRAINING: {name}")
    print(f"Epochs={epochs} | Patience={patience} | InitLR={lr} | "
          f"GradClip=1.0 | Optimizer=Adam")
    print(f"{'='*70}")
    prev_lr = lr

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss, batches = 0.0, 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        avg_train = epoch_loss / batches
        train_losses.append(avg_train)

        # Validation phase
        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(test_X_tensor), test_y_tensor).item()
        test_losses.append(test_loss)
        scheduler.step(test_loss)

        # Early-stopping bookkeeping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state     = copy.deepcopy(model.state_dict())
            patience_left  = 0
        else:
            patience_left += 1

        # One-line log per epoch (kept for thesis Output 5.2)
        cur_lr = optimizer.param_groups[0]['lr']
        new_best = "  <-- new best" if patience_left == 0 else ""
        lr_drop  = f"  >> LR halved: {prev_lr:.6f} -> {cur_lr:.6f}" if cur_lr < prev_lr else ""
        print(f"[{name}] Epoch {epoch+1:3d}/{epochs} | "
              f"train={avg_train:.6f} | val={test_loss:.6f} | "
              f"lr={cur_lr:.6f} | patience={patience_left:2d}/{patience}"
              f"{new_best}{lr_drop}")
        prev_lr = cur_lr

        if patience_left >= patience:
            print(f"{'-'*70}")
            print(f">> EARLY STOPPING at epoch {epoch+1} "
                  f"(no improvement for {patience} epochs)")
            print(f">> Best val loss seen: {best_test_loss:.6f}")
            break

    # Restore the best weights and produce final predictions
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(test_X_tensor).numpy()

    print(f"{'-'*70}")
    print(f"[{name}] DONE | best val loss = {best_test_loss:.6f} | "
          f"epochs run = {len(train_losses)}/{epochs}")
    print(f"{'='*70}")
    return predictions, train_losses, test_losses


def calc_mape(y_true, y_pred):
    """Mean absolute percentage error."""
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


# Main training loop
input_size     = train_X.shape[2]            # 10
stock_tickers  = df['ticker'].unique()       # 10 tickers

# `runs` collects per-stock RMSE for every (model, run) so we can compute
# mean and std at the end.
runs = {name: {t: [] for t in stock_tickers} for name in ('LSTM', 'RNN', 'GRU', 'CNN')}
overall_rmse = {name: [] for name in ('LSTM', 'RNN', 'GRU', 'CNN')}

# `best` keeps the best-RMSE run for each model its predictions, train/valloss curves, and the on-disk .pth file
best = {name: dict(rmse=float('inf'), preds=None,
                   train_losses=None, test_losses=None,
                   pth=f"{name.lower()}_model.pth")
        for name in ('LSTM', 'RNN', 'GRU', 'CNN')}

MODEL_CLASSES = {
    'LSTM': LSTMModel,
    'RNN':  RNNModel,
    'GRU':  GRUModel,
    'CNN':  CNN1DModel,
}

# Train each neural model NUM_RUNS times -- keeps the best by RMSE.
# Random init means a single run can be lucky or unlucky.
for run in range(NUM_RUNS):
    print(f"\n{'#'*60}\nRUN {run + 1}/{NUM_RUNS}\n{'#'*60}")

    train_loader = DataLoader(
        TensorDataset(train_X_tensor, train_y_tensor),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    run_preds = {}    # model name -> predictions for this run

    for name, ModelClass in MODEL_CLASSES.items():
        # CNN constructor needs seq_len, the others don't.
        model = (ModelClass(input_size=input_size, seq_len=WINDOW_SIZE)
                 if name == 'CNN' else ModelClass(input_size=input_size))
        preds, train_l, test_l = train_model(
            model, name, train_loader, test_X_tensor, test_y_tensor,
            EPOCHS, LEARNING_RATE, PATIENCE,
        )
        run_rmse = np.sqrt(mean_squared_error(test_y, preds))
        overall_rmse[name].append(run_rmse)
        run_preds[name] = preds

        # Save this run as the best if it beat the previous best
        if run_rmse < best[name]['rmse']:
            best[name].update(
                rmse=run_rmse, preds=preds.copy(),
                train_losses=train_l, test_losses=test_l,
            )
            torch.save(model.state_dict(), best[name]['pth'])

    # Per-stock RMSE for this run 
    start_idx = 0
    for ticker in stock_tickers:
        n_test = len(test_dates[ticker])
        s_actual = test_y[start_idx:start_idx + n_test]
        for name in MODEL_CLASSES:
            slice_preds = run_preds[name][start_idx:start_idx + n_test]
            runs[name][ticker].append(np.sqrt(mean_squared_error(s_actual, slice_preds)))
        start_idx += n_test


# Random Forest baseline -- can't read sequences, so flatten (20, 10) into 200.
print(f"\n{'='*50}\nTraining Random Forest...\n{'='*50}")
rf_train_X = train_X.reshape(train_X.shape[0], -1)
rf_test_X  = test_X.reshape(test_X.shape[0], -1)
rf_model = RandomForestRegressor(
    n_estimators=200, max_depth=20,
    min_samples_split=5, min_samples_leaf=2,
    random_state=42, n_jobs=-1,
)
rf_model.fit(rf_train_X, train_y)
rf_preds = rf_model.predict(rf_test_X)

with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

rf_rmse = np.sqrt(mean_squared_error(test_y, rf_preds))
rf_mae  = mean_absolute_error(test_y, rf_preds)
rf_mape = calc_mape(test_y, rf_preds)


# Final summary
print(f"\n{'='*60}\nFINAL RESULTS (averaged over {NUM_RUNS} runs)\n{'='*60}")
for name in MODEL_CLASSES:
    mean = np.mean(overall_rmse[name]); std = np.std(overall_rmse[name])
    bp   = best[name]['preds']
    print(f"{name}: RMSE = {mean:.6f} +/- {std:.6f} "
          f"(best: {best[name]['rmse']:.6f}) | "
          f"MAE={mean_absolute_error(test_y, bp):.6f} | "
          f"MAPE={calc_mape(test_y, bp):.2f}%")
print(f"Random Forest: RMSE={rf_rmse:.6f} | MAE={rf_mae:.6f} | MAPE={rf_mape:.2f}%")


# Build per-stock results table -> model_results.json
results = []
start_idx = 0
for ticker in stock_tickers:
    n_test   = len(test_dates[ticker])
    s_actual = test_y[start_idx:start_idx + n_test]
    s_rf     = rf_preds[start_idx:start_idx + n_test]
    rf_stock = np.sqrt(mean_squared_error(s_actual, s_rf))

    row = {'ticker': ticker}
    rmse_dict = {}     # which model is best for this stock?
    for name in MODEL_CLASSES:
        avg = float(np.mean(runs[name][ticker]))
        std = float(np.std(runs[name][ticker]))
        bp  = best[name]['preds'][start_idx:start_idx + n_test]
        row[f"{name.lower()}_rmse_avg"]  = round(avg, 6)
        row[f"{name.lower()}_rmse_std"]  = round(std, 6)
        row[f"{name.lower()}_rmse_best"] = round(float(np.sqrt(mean_squared_error(s_actual, bp))), 6)
        row[f"{name.lower()}_mae_best"]  = round(float(mean_absolute_error(s_actual, bp)), 6)
        rmse_dict[name] = avg
    rmse_dict['Random Forest'] = rf_stock
    row['rf_rmse']  = round(float(rf_stock), 6)
    row['rf_mae']   = round(float(mean_absolute_error(s_actual, s_rf)), 6)
    row['lstm_wins'] = f"{sum(1 for l, r in zip(runs['LSTM'][ticker], runs['RNN'][ticker]) if l < r)}/{NUM_RUNS}"
    row['winner']   = min(rmse_dict, key=rmse_dict.get)
    results.append(row)
    start_idx += n_test

with open('model_results.json', 'w') as f:
    json.dump(results, f, indent=2)


# Save loss history (api/loss_history)
loss_history = {
    name: {'train': best[name]['train_losses'], 'val': best[name]['test_losses']}
    for name in MODEL_CLASSES
}
with open('loss_history.json', 'w') as f:
    json.dump(loss_history, f)
print("Loss history saved to loss_history.json")


# Plot loss curves 4 panels in a 2x2 grid
colors = {'LSTM': '#2196F3', 'RNN': '#F44336', 'GRU': '#9C27B0', 'CNN': '#FF5722'}
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, name in zip(axes.flatten(), MODEL_CLASSES):
    ax.plot(best[name]['train_losses'], label='Train',
            linewidth=1.5, color=colors[name])
    ax.plot(best[name]['test_losses'],  label='Test',
            linewidth=1.5, color='#FF9800')
    ax.set_title(f'{name} Loss Curve', fontsize=14)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (MSE)')
    ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig('loss_curves.png', dpi=150, bbox_inches='tight')
plt.close()


# Bar chart of average RMSE across 5 models
fig, ax = plt.subplots(figsize=(10, 5))
names_all = ['LSTM', 'RNN', 'GRU', 'CNN', 'Random Forest']
rmses     = [np.mean(overall_rmse['LSTM']), np.mean(overall_rmse['RNN']),
             np.mean(overall_rmse['GRU']),  np.mean(overall_rmse['CNN']), rf_rmse]
stds      = [np.std(overall_rmse['LSTM']),  np.std(overall_rmse['RNN']),
             np.std(overall_rmse['GRU']),   np.std(overall_rmse['CNN']), 0]
bar_cols  = ['#2196F3', '#F44336', '#9C27B0', '#FF5722', '#4CAF50']

x = np.arange(len(names_all))
bars = ax.bar(x, rmses, 0.45, yerr=stds, capsize=5, color=bar_cols, alpha=0.85)
ax.set_ylabel('RMSE')
ax.set_title(f'Model RMSE Comparison (Average of {NUM_RUNS} Runs +/- Std)', fontsize=14)
ax.set_xticks(x); ax.set_xticklabels(names_all); ax.grid(True, axis='y')
for bar, val, std in zip(bars, rmses, stds):
    label = f'{val:.4f}+/-{std:.4f}' if std > 0 else f'{val:.4f}'
    ax.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5), textcoords="offset points",
                ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()


# One prediction-vs-actual plot per stock 10 PNG
pred_colors = {'LSTM': '#2196F3', 'RNN': '#F44336', 'GRU': '#9C27B0',
               'CNN': '#FF5722', 'Random Forest': '#4CAF50'}

start_idx = 0
for ticker in stock_tickers:
    n_test = len(test_dates[ticker])
    sliced = {name: best[name]['preds'][start_idx:start_idx + n_test]
              for name in MODEL_CLASSES}
    sliced['Random Forest'] = rf_preds[start_idx:start_idx + n_test]
    s_actual = test_y[start_idx:start_idx + n_test]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(range(n_test), s_actual, label='Actual', color='black', linewidth=2)
    for name, preds in sliced.items():
        ax.plot(range(n_test), preds, label=name,
                color=pred_colors[name], alpha=0.8, linewidth=1.2)
    ax.set_title(f'{ticker} - Price Prediction Comparison (Normalized)', fontsize=14)
    ax.set_xlabel('Trading Days'); ax.set_ylabel('Normalized Close Price')
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    fig.savefig(f'prediction_{ticker.replace(".", "_")}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    start_idx += n_test


# Save held-out predictions to MySQL
print("Saving predictions to MySQL...")
conn = mysql.connector.connect(**DB)
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS model_predictions")
cursor.execute("""
    CREATE TABLE model_predictions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        ticker VARCHAR(10),
        date DATE,
        actual DECIMAL(12,8),
        lstm_pred DECIMAL(12,8),
        rnn_pred DECIMAL(12,8),
        gru_pred DECIMAL(12,8),
        cnn_pred DECIMAL(12,8),
        rf_pred DECIMAL(12,8)
    )
""")

start_idx = 0
for ticker in stock_tickers:
    n_test = len(test_dates[ticker])
    for i in range(n_test):
        cursor.execute(
            "INSERT INTO model_predictions "
            "(ticker, date, actual, lstm_pred, rnn_pred, gru_pred, cnn_pred, rf_pred) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (ticker, str(test_dates[ticker][i])[:10],
             float(test_y[start_idx + i]),
             float(best['LSTM']['preds'][start_idx + i]),
             float(best['RNN']['preds'][start_idx + i]),
             float(best['GRU']['preds'][start_idx + i]),
             float(best['CNN']['preds'][start_idx + i]),
             float(rf_preds[start_idx + i])),
        )
    start_idx += n_test

conn.commit(); cursor.close(); conn.close()
print("Predictions saved to model_predictions table.")


# 5-day recursive forecast.
# For each stock: predict day +1, write that prediction into the next window,
# predict day +2, and so on. The other 9 features for the new day are copied
# from the previous day.
print("Generating 5-day future forecasts...")

conn = mysql.connector.connect(**DB)
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS model_forecasts")
cursor.execute("""
    CREATE TABLE model_forecasts (
        id INT AUTO_INCREMENT PRIMARY KEY,
        ticker VARCHAR(10),
        day_ahead INT,
        forecast_date DATE,
        actual DECIMAL(12,8),
        lstm_pred DECIMAL(12,8),
        rnn_pred DECIMAL(12,8),
        gru_pred DECIMAL(12,8),
        cnn_pred DECIMAL(12,8),
        rf_pred DECIMAL(12,8)
    )
""")

# Reload best saved weights -- eval mode turns off dropout / batchnorm noise
loaded = {}
for name, ModelClass in MODEL_CLASSES.items():
    m = (ModelClass(input_size=input_size, seq_len=WINDOW_SIZE)
         if name == 'CNN' else ModelClass(input_size=input_size))
    m.load_state_dict(torch.load(best[name]['pth']))
    m.eval()
    loaded[name] = m

for ticker in stock_tickers:
    stock_data   = df[df['ticker'] == ticker].sort_values('date')
    all_features = stock_data[FEATURE_COLS].values
    all_dates    = stock_data['date'].values

    # Go back 25 days from the end: 20-day window + 5 days of ground truth
    start_pt   = len(all_features) - WINDOW_SIZE - 5
    input_20   = all_features[start_pt:start_pt + WINDOW_SIZE]
    actual_5   = all_features[start_pt + WINDOW_SIZE:start_pt + WINDOW_SIZE + 5, 0]
    dates_5    = all_dates[start_pt + WINDOW_SIZE:start_pt + WINDOW_SIZE + 5]

    # Each model owns its own rolling window because predictions diverge
    windows   = {name: input_20.copy() for name in MODEL_CLASSES}
    windows['RF'] = input_20.copy()
    forecasts = {name: [] for name in list(MODEL_CLASSES) + ['RF']}

    for day in range(5):
        # Predict day t with each neural model add batch dim -> (1, 20, 10)
        with torch.no_grad():
            for name in MODEL_CLASSES:
                x = torch.FloatTensor(windows[name][-WINDOW_SIZE:]).unsqueeze(0)
                forecasts[name].append(loaded[name](x).item())
        # Predict with RF (needs flat 200-vector)
        rf_pred = rf_model.predict(windows['RF'][-WINDOW_SIZE:].reshape(1, -1))[0]
        forecasts['RF'].append(rf_pred)

        # Build next day's row: copy yesterday, replace close_norm
        for name in list(MODEL_CLASSES) + ['RF']:
            new_row = windows[name][-1].copy()
            new_row[0] = forecasts[name][-1]
            windows[name] = np.vstack([windows[name], new_row])

    # Persist the 5 forecast days
    for day in range(5):
        cursor.execute(
            "INSERT INTO model_forecasts "
            "(ticker, day_ahead, forecast_date, actual, lstm_pred, rnn_pred, "
            " gru_pred, cnn_pred, rf_pred) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (ticker, day + 1, str(dates_5[day])[:10], float(actual_5[day]),
             float(forecasts['LSTM'][day]), float(forecasts['RNN'][day]),
             float(forecasts['GRU'][day]),  float(forecasts['CNN'][day]),
             float(forecasts['RF'][day])),
        )

conn.commit(); cursor.close(); conn.close()
print("5-day forecasts saved to model_forecasts table.")
print("\nAll done.")
