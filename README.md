# NiftyML — Earnings Drift Intelligence Platform

## The idea
After a genuine earnings beat, institutional money takes 10–15 days to fully position.
Retail can get in front of that wave on Day 1 and ride it.
ML's job is to rank which gap-ups are real vs noise.

## Project structure
```
niftyml/
├── pipeline.py          ← Run this once to train + save the model
├── app.py               ← Flask app (two pages only)
├── requirements.txt
├── src/
│   ├── earnings/
│   │   ├── universe.py        ← NSE stock list
│   │   ├── data_loader.py     ← Download + event detection
│   │   ├── features.py        ← Feature engineering (10d target)
│   │   ├── model.py           ← XGBoost regressor + walk-forward CV
│   │   ├── backtest.py        ← Long-only, 4% stop, top-25% filter
│   │   └── dashboard_data.py  ← Data builder for /earnings page
│   └── stock/
│       └── engine.py          ← Per-stock deep dive for /stock page
├── templates/
│   ├── base.html              ← Aurora Terminal design system
│   ├── earnings.html          ← Main product page
│   └── stock.html             ← Per-stock analyser
├── models/                    ← Saved model outputs (git-ignored)
├── reports/                   ← Backtest JSON (git-ignored)
└── data/                      ← Cached downloads (git-ignored)
```

## Quickstart
```bash
pip install -r requirements.txt

# Step 1: Train the model (takes ~10 min first run — downloads data)
python pipeline.py

# Step 2: Run the app
python app.py
# → http://localhost:5000/earnings
```

## To retrain with fresh data
```bash
python pipeline.py --fresh
```

## Pages
- `/earnings`     — Earnings Intelligence scanner (main product)
- `/stock/<T>`    — Per-stock earnings drift deep dive
- `/api/earnings` — JSON API for the earnings data
- `/api/stock/<T>`— JSON API for per-stock analysis

## Why this works (and the old one didn't)
| Old pipeline | New pipeline |
|---|---|
| Predicts next-day candle direction | Predicts 10d post-earnings drift |
| 49.85% accuracy (coin flip) | Lift metric — top 25% vs baseline |
| Goes short on every 0 signal | Long-only, 4% hard stop |
| models/ always empty | Model saved to models/ automatically |
| Nifty 50 — most efficient market | Mid/small caps — analyst coverage gaps |
