# Steam Game Rating Predictor

A Streamlit app that wraps your notebook model for predicting a game's share of
positive Steam reviews (`pct_pos_total`) from pre-release metadata, plus an
interactive EDA dashboard.

## Files

| File | Purpose |
|------|---------|
| `pipeline.py` | The whole model as one picklable `GameRatingPipeline` class: feature engineering, MultiLabelBinarizer (genres + categories), TargetEncoder (developer/publisher), and the sklearn pipeline (Lasso by default). Imported by both the trainer and the app. |
| `train_model.py` | Run once to fit on your CSV and write `game_rating_model.pkl`. |
| `app.py` | The Streamlit app: single prediction, batch CSV prediction, EDA dashboard. |
| `requirements.txt` | Dependencies. |

## Setup

```bash
pip install -r requirements.txt
```

## 1. Train the model (once)

Point it at your cleaned dataset (the table from the top of the notebook):

```bash
python train_model.py --data Final_clean_df.csv --out game_rating_model.pkl
```

Options: `--model {lasso,ridge,elasticnet}` (default `lasso`, the notebook's best),
`--test-size 0.2`. Training uses the same **time-based split** as the notebook
(earliest 80% train, latest 20% test) and prints holdout RMSE / MAE / R².

## 2. Run the app

```bash
streamlit run app.py
```

It auto-loads `game_rating_model.pkl` if present. If not, the sidebar lets you
upload a `.pkl` or train one in-app by uploading the CSV.

## What the app does

- **🎯 Single Prediction** — a form for one game's pre-release metadata; returns the
  expected positive-review % and a sentiment band.
- **📦 Batch (CSV)** — upload a table in the training format, score every row,
  download results. If the CSV includes `pct_pos_total`, it also reports MAE/RMSE
  and a predicted-vs-actual plot.
- **📊 EDA Dashboard** — upload the dataset to explore genre influence on ratings
  over time (per-year lines + genre×year heatmap), genre rankings/distributions,
  release trends, price tiers, seasonality, and the overall rating distribution.
  Year-range and genre filters are interactive.

## Input format

The pipeline expects the raw columns from the start of the notebook. Key ones:
`release_date`, `genres`, `categories`, `developers`, `publishers`,
`full_audio_languages` (list-like or stringified lists), `about_the_game`,
`mac`, `linux`, `os_count`, `lang_count`, `lang_group`, `price`, `year`, the
`has_*` content/language flags, and `pct_pos_total` (target, only for training).
Missing optional columns are filled with sensible defaults, so partial single
records still work.

## Note vs. the original notebook

In the notebook the `MultiLabelBinarizer` cell ran after the genres/categories
loop, so only **categories** were binarized and the genre columns were never
added. This pipeline fixes that and binarizes **both** genres and categories,
which is the intended behaviour and what the dashboard's genre analysis relies
on. Because of this the feature set differs slightly from the notebook, so exact
metrics may shift a little, but the modelling approach is identical.
