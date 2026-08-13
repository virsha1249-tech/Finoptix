# FinOptiX — S&P 500 Portfolio Optimization with Multi-Model Forecasting

FinOptiX is a Jupyter notebook pipeline that blends classical finance theory with machine learning to build a risk-constrained, max-Sharpe equity portfolio from a sector-diversified subset of S&P 500 stocks.

## Objective

Combine four independent expected-return signals — one econometric, three ML-based — into a single composite forecast per stock, then feed that forecast into a mean-variance optimizer (with an explicit market-beta ceiling) to construct a portfolio and backtest it against the S&P 500 benchmark.

## Pipeline Overview

1. **Data acquisition** — Downloads multi-year daily OHLCV data for a ~30–40 stock, sector-diversified S&P 500 subset (plus the S&P 500 index itself as benchmark) via `yfinance`. A commented-out Wikipedia scrape is included if you want the full ~500-ticker universe instead.

2. **Fama-French 5-Factor model (OLS)** — Pulls the daily Fama-French factors (Mkt-RF, SMB, HML, RMW, CMA, RF) from Kenneth French's data library and regresses each stock's excess return against them, producing factor loadings (betas) and a factor-implied expected return per stock.

3. **Feature engineering** — Builds technical features per stock (momentum, volatility, moving averages, Bollinger bands, volume-price correlation, lagged returns) with a forward 21-trading-day return as the prediction target.

4. **XGBoost & Random Forest** — Trained per-stock on the engineered features to predict forward 21-day return. The two models' predictions are averaged into a single "ML expected return" signal.

5. **LSTM** — A sequence model trained per stock on a 30-day sliding window of raw OHLCV data to capture temporal dependencies the row-wise tree models can't, producing a third expected-return signal. (Flagged as the most compute-heavy step in the notebook.)

6. **ARIMA** — Fit per-stock on the closing price series (with ADF stationarity testing and a small grid search over `(p,d,q)` by AIC) to produce a fourth, purely time-series-based expected-return signal.

7. **Signal blending & portfolio optimization** — All four signals are z-scored and averaged into a composite expected-return score. The top 15 stocks by composite score are selected, and `PyPortfolioOpt` runs max-Sharpe mean-variance optimization twice:
   - **Unconstrained** — free to select any beta exposure.
   - **Beta-constrained** — adds a hard linear constraint (`portfolio beta ≤ 0.85`) so the optimizer can't simply lever up on market risk to boost Sharpe.

   Per-stock betas vs. the S&P 500 are computed via OLS to support the constraint.

8. **Backtest** — The beta-constrained portfolio's daily returns are tracked against the S&P 500 over a trailing 2-year window, reporting cumulative return, CAGR outperformance, annualized volatility, Sharpe ratio, and realized portfolio beta (including a rolling 60-day beta plot to check stability of risk exposure over time).

## Key Outputs

- Heatmap of Fama-French factor loadings across the stock universe
- Bar chart ranking stocks by ML-predicted forward return
- Distribution of LSTM out-of-sample R² across tickers
- ARIMA forecast diagnostic plot
- Composite multi-model score ranking (top 15 stocks selected for the portfolio)
- Unconstrained vs. beta-constrained performance comparison table (expected return, volatility, Sharpe, beta)
- Final optimized portfolio weights (bar chart)
- Portfolio vs. benchmark cumulative return / CAGR outperformance
- Rolling 60-day portfolio beta vs. the 0.85 target ceiling

## Requirements

```
yfinance
pandas-datareader
PyPortfolioOpt
xgboost
statsmodels
tensorflow
scikit-learn
seaborn
plotly
pandas
numpy
matplotlib
```

Install via the notebook's first cell:
```
pip install -q yfinance pandas-datareader PyPortfolioOpt xgboost statsmodels tensorflow scikit-learn seaborn plotly
```

## How to Run

1. Open the notebook (Google Colab or local Jupyter with the packages above installed).
2. Run cells top to bottom. Adjust `start_date` / `end_date` and the `tickers` list as needed.
3. The LSTM section (Section 5) is the slowest step — reduce `epochs` or the ticker list for faster iteration.
4. Final beta ceiling (`BETA_CEILING`) and portfolio size (`TOP_N`) are configurable constants in Sections 5 and 7 respectively.

## Notes & Caveats

- The 40-name ticker list is a hand-picked, sector-diversified subset rather than the full S&P 500, chosen to keep per-ticker model training (especially LSTM) tractable. A commented-out cell lets you swap in the full index.
- The beta-constrained portfolio is expected to show lower expected return and Sharpe than the unconstrained one — this is the cost of the risk constraint, not a modeling error.
- All expected-return signals are historical/backtest-based estimates and are not investment advice.
