# Stock Price Prediction Using LSTM

## The Idea

Predicting stock prices has long been the holy grail of quantitative finance. The challenge is inherently sequential — what happened yesterday matters today, and what happened last week can still cast a shadow on tomorrow's price. Traditional regression models miss this. That's exactly where **Long Short-Term Memory (LSTM)** networks come in.

LSTMs are a refined class of Recurrent Neural Networks (RNNs) designed to remember patterns over long sequences without the classic "vanishing gradient" problem. This project applies that architecture to Microsoft (`MSFT`) stock data — 10 years of adjusted closing prices — to see how well a deep sequence model can anticipate the next day's price.

---

## Architecture & Framework

### Data Pipeline
- **Source:** Yahoo Finance via `yfinance`, pulling 10 years of `MSFT` adjusted close prices
- **Preprocessing:** MinMax scaling to compress all values into the `[0, 1]` range — a critical step because LSTMs are sensitive to input magnitudes
- **Train/Test Split:** 80% training, 20% testing (no shuffling — time order is sacred)

### Sequence Construction
The model is fed **60-day rolling windows** of historical prices as input sequences (`X`), with the next day's price as the target (`y`). This window-based approach teaches the model to reason over a rolling month and a half of context at every prediction step. The data is reshaped into 3D tensors of shape `(samples, 60 timesteps, 1 feature)` as required by Keras.

### Model Architecture
```
Layer 1: LSTM (100 units) — return_sequences=True
Layer 2: LSTM (100 units) — return_sequences=False
Layer 3: Dense (25 units)
Layer 4: Dense (1 unit) → Price prediction output
```
- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error
- **Training:** 3 epochs, batch size of 1

The two stacked LSTM layers allow the network to learn both short-range (recent days) and longer-range (multi-week) dependencies in the price sequence.

---

## Key Results

Training loss dropped steadily from `0.0012` to `0.00043` across three epochs — smooth and stable, with no signs of divergence.

On the test set, the regression metrics look solid:

| Metric | Value |
|--------|-------|
| RMSE | Low (visually, the predicted curve hugs the actual curve) |
| MAE | Low |
| R² | High |
| MAPE | Reasonable percentage error |

But here's where it gets honest.

### The Directional Reality Check
When the same predictions are evaluated not on *how close* they are, but on *whether they called the direction correctly* (up or down), the numbers look very different:

| Directional Metric | Value |
|--------------------|-------|
| Accuracy | ~49.7% |
| Precision | ~51.89% |
| Recall | ~52.29% |
| F1 Score | ~52.09% |

Barely above a coin flip.

---

## Why This Happens — The "Lag" Catch

The LSTM achieves low visual error because tomorrow's price is almost always close to today's. The model learns this shortcut and essentially echoes the recent past — a behavior sometimes called "naive forecasting." It looks great on an RMSE chart but fails when asked *"should I buy or sell?"*

This is one of the most important lessons in applied ML for finance: **low prediction error ≠ useful trading signal.**

---

## Key Takeaways

1. **Stacked LSTMs are a strong baseline** for sequential price modeling — they converge well and generalize without obvious overfitting.
2. **The train-test gap is healthy** (near zero), meaning the architecture itself is sound. The problem isn't overfitting; it's that absolute price prediction is the wrong objective for trading.
3. **RMSE and R² are misleading metrics here** — directional accuracy is what actually matters for generating alpha.
4. **The 60-day window is a sensible heuristic** but isn't optimized. Different lookback windows could meaningfully change what the network learns.

---

## What Would Make This Production-Ready

To take this from a compelling proof-of-concept to something deployable:

- **Switch the target to log returns**, not price levels. Returns are stationary; prices aren't.
- **Expand the feature space** — volume, RSI, MACD, macro indicators. Raw closing prices alone are a thin information diet.
- **Activate dropout regularization** and implement proper validation splits with early stopping.
- **Use walk-forward validation** instead of a single train/test split to simulate real trading conditions.
- **Build a data pipeline** (AWS Lambda / Airflow) that fetches daily close prices, applies the saved `MinMaxScaler` (serialized as `.pkl`), and feeds the rolling 60-day window into the model at market close.
- **Monitor for data drift** and schedule monthly retraining as market regimes shift.

---

## Final Remarks

This project succeeds at what it sets out to do: demonstrate that a stacked LSTM can learn the structure of a stock price sequence and track it visually with impressive fidelity. What it honestly reveals, though, is the gap between *forecasting accuracy* and *trading utility* — a distinction that separates academic backtests from real alpha generation. The architecture is the right foundation. The next step is asking the right question of it.
