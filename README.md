# Flight Delay Prediction — Uncertainty-Aware & Network-Informed

This repository predicts **departure delay (binary + minutes regression)** and estimates **uncertainty** using **Conformal Prediction**, while modeling **airport-to-airport delay propagation** through a lightweight **graph-based feature pipeline**.

### Why this is different
- **Uncertainty you can trust**: Distribution-free **Conformal Prediction** gives calibrated probability intervals.
- **Network effects**: Builds an **airport graph** from historical delays to capture **propagation**.
- **Strong baselines & clarity**: Clean, reproducible ML pipeline with `sklearn`, `xgboost`, and explainability via `SHAP`.

### Data
Use any DOT/BTS or Kaggle airline on-time dataset (e.g., 2015) saved as CSV to `data/raw/flights.csv` with standard columns (Year, Month, DayOfWeek, Airline, Origin, Dest, DepDelay, ArrDelay, WeatherDelay, etc.).

### Quickstart
```bash
# 1) Create environment
python -m venv .venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Run baseline training (define CSV path or use default)
python src/train_baseline.py --csv data/raw/flights.csv --target arrival_delay_minutes

# 3) Add conformal prediction
python src/train_conformal.py --csv data/raw/flights.csv --target arrival_delay_minutes

# 4) Network-informed features
python src/features/build_airport_graph.py --csv data/raw/flights.csv --out data/processed/graph_features.parquet
python src/train_network_augmented.py --csv data/raw/flights.csv --graph data/processed/graph_features.parquet
```

### Project structure
```
data/
  raw/                # place original CSV(s) here
  processed/          # generated features & splits
notebooks/            # EDA & experiments
src/
  features/           # feature builders (calendar, weather join, graph features)
  models/             # model definitions
  utils/              # helpers (io, metrics)
  train_baseline.py   # classic ML baselines
  train_conformal.py  # conformal wrappers for calibrated uncertainty
  train_network_augmented.py  # baseline + graph features
configs/
  baseline.yaml       # config for reproducibility
reports/
  figures/            # EDA plots
.github/workflows/
  ci.yml              # Lint & smoke tests
```

### Novelty ideas (extendable)
- **Airport graph features** (delay centrality, PageRank, rolling in-degree delay).
- **Conformal prediction** for calibrated intervals on delay minutes / on-time probability.
- **Survival modelling** for *time-to-arrival* beyond schedule (Kaplan–Meier / Cox) [optional].
- **Fairness & drift checks** across airlines/airports and time.

### License
MIT
