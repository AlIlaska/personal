# Steam pre-release rating predictor — Streamlit app

Predicts the **percentage of positive reviews** a Steam game will receive, from
metadata known *before* release. It packages the notebook's final model (a
**Lasso** regression on a logit-transformed target) into one self-contained,
pickled pipeline, and adds a prediction UI, batch scoring, SHAP explanations, and
an EDA dashboard.

The `price` column is never used — neither for prediction nor in the dashboard.

## Files

| File | What it does |
|------|--------------|
| `pipeline_utils.py` | The end-to-end pipeline + custom `SteamFeatureEngineer` (all the notebook's feature engineering: list parsing, dev/publisher target encoding, genre/category multi-label binarizing, date features, TF-IDF). **Must stay importable** — the pickle references these classes. |
| `train_model.py` | Trains on your CSV and writes `artifacts/` (model + metadata + EDA data). |
| `app.py` | The Streamlit app (Predict + EDA Dashboard pages). |
| `explain.py` | SHAP / feature-weight helpers used by the app. |
| `generate_synthetic_data.py` | Makes a fake `clean_data_v1.csv` so you can demo the app without your real data. |
| `requirements.txt` | Dependencies. |

## Setup

```bash
pip install -r requirements.txt
```

## 1. Train (creates the pickled pipeline)

Put your real `clean_data_v1.csv` (same format as the top of the notebook) next
to these files, then:

```bash
python train_model.py --data clean_data_v1.csv
```

This prints the time-split test metrics (reproducing the notebook's Lasso
results), then refits on all rows and writes:

```
artifacts/
  model.pkl          # call model.predict(raw_dataframe) -> predicted % positive
  metadata.pkl       # dropdown choices + defaults for the input form
  background.pkl      # sample used by SHAP
  eda_data.parquet   # cleaned data for the dashboard (price dropped)
```

**No real data yet?** Generate synthetic data to try the UI end to end:

```bash
python generate_synthetic_data.py
python train_model.py --data clean_data_v1.csv
```

(Synthetic predictions are meaningless — they only prove the app runs.)

## 2. Run the app

```bash
streamlit run app.py
```

### Predict page
- **Single game** — inputs grouped into expanders: *Basic Information*, *Genres
  and Tags*, *Technical Specifications*, *Marketing Data*. Returns the predicted
  % positive plus a SHAP explanation: the top 3 🟢 factors raising the score and
  top 3 🔴 lowering it, the Lasso feature weights (the model is linear), and
  global SHAP importance.
- **Batch (CSV)** — upload a CSV in the source format; every row is scored,
  downloadable as CSV, and you can drill into any row's explanation.

### EDA Dashboard
KPIs plus rating distribution, releases per year, average rating by genre, a
genre × year heatmap and genre rating trend over time, platform / language /
content-descriptor effects, and top publishers. Year-range and genre filters
live in the sidebar.

## Notes / design choices

- The whole thing is one `TransformedTargetRegressor` → `Pipeline([feature
  engineer, preprocessor, LassoCV])`. Feeding it a raw-format row does all
  parsing, encoding and TF-IDF internally, so deployment is just
  `model.predict(df)`.
- Developer/publisher **target encoding** is fit inside the pipeline. Because the
  target is logit-transformed upstream, the encoding lives in logit space rather
  than on the raw 0–100 scale as in the notebook. The downstream scaler
  normalizes it and the transform is monotonic, so predictions are equivalent;
  this just makes the object cleanly picklable and leak-safe.
- For a linear model, SHAP values equal `coef · (x − E[x])`; the app uses
  `shap.LinearExplainer` and rescales to approximate percentage points via the
  logit derivative (same approach as the notebook), with a closed-form fallback
  if `shap` isn't available.
