"""
Обучите модель для прогнозирования рейтинга игры в Steam до её выхода и сохраните готовый к использованию конвейер (pipeline).

Использование:
python train_model.py --data clean_data_v1.csv

Результаты (сохраняются в папку ./artifacts):
model.pkl        - Объект TransformedTargetRegressor, объединяющий полный конвейер обработки. 
Вызов model.predict(raw_dataframe) возвращает прогнозируемую долю положительных отзывов (в %). 
metadata.pkl     - Списки признаков, значения по умолчанию и варианты выбора для формы приложения,
а также фоновая выборка данных для анализа методом SHAP. 
eda_data.parquet - Очищенный набор данных для дашборда разведочного анализа (EDA).

Итоговая модель воспроизводит подход, выбранный в ноутбуке (регрессия Lasso с целевой переменной, преобразованной через логит-функцию).
Сначала модель оценивается на выборке, разделенной по временному признаку в пропорции 80/20 (для воспроизведения метрик из ноутбука),
а затем переобучается на всех доступных данных для внедрения в эксплуатацию.
"""

import argparse
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit, expit

from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from pipeline_utils import (
    SteamFeatureEngineer, build_preprocessor, to_logit, from_logit,
    TARGET, BASE_NUMERICAL, CATEGORICAL_FEATURES, LIST_COLS, parse_list,
)

warnings.filterwarnings("ignore")
ART = "artifacts"


# --------------------------------------------------------------------------- #
def load_and_clean(path):
    df = pd.read_csv(path)
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    if "year" not in df.columns and "release_date" in df.columns:
        df["year"] = df["release_date"].dt.year
    # notebook filters
    if "year" in df.columns:
        df = df[df["year"] > 2008]
    if "price" in df.columns:
        df = df[df["price"] >= 0]
    df = df[df[TARGET].notna()].reset_index(drop=True)
    # ensure list columns are real lists for downstream use
    for c in LIST_COLS:
        if c in df.columns:
            df[c] = df[c].apply(parse_list)
    return df


def build_model(X_fit, y_fit_logit):
    """Build a fresh end-to-end model (feature lists discovered from a peek-fit)."""
    fe_peek = SteamFeatureEngineer().fit(X_fit, y_fit_logit)
    preproc = build_preprocessor(
        fe_peek.numerical_features_, fe_peek.categorical_features_, fe_peek.text_feature_
    )
    inner = Pipeline([
        ("fe", SteamFeatureEngineer()),
        ("prep", preproc),
        ("model", LassoCV(alphas=np.logspace(-3, 2, 20), cv=KFold(5, shuffle=True, random_state=42),
                          max_iter=3000, n_jobs=-1, tol=1e-3)),
    ])
    return TransformedTargetRegressor(regressor=inner, func=to_logit, inverse_func=from_logit)


def report(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  {name:6s}  RMSE={rmse:7.4f}  MAE={mae:7.4f}  R2={r2:7.4f}")
    return dict(rmse=rmse, mae=mae, r2=r2)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="clean_data_v1.csv")
    ap.add_argument("--out", default=ART)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"Loading {args.data} ...")
    df = load_and_clean(args.data)
    print(f"  rows after cleaning: {len(df)}")

    # ---- time-based split (reproduces notebook metrics) ------------------- #
    df_sorted = df.sort_values("release_date").reset_index(drop=True)
    split = int(len(df_sorted) * 0.8)
    train, test = df_sorted.iloc[:split], df_sorted.iloc[split:]
    y_train, y_test = train[TARGET].values, test[TARGET].values

    print("\nTime-based 80/20 evaluation (Lasso, logit target):")
    eval_model = build_model(train, to_logit(y_train))
    eval_model.fit(train, y_train)            # TTR transforms y internally
    report("train", y_train, eval_model.predict(train))
    metrics = report("test ", y_test, eval_model.predict(test))

    # ---- refit on ALL data for deployment --------------------------------- #
    print("\nRefitting on all rows for deployment ...")
    model = build_model(df, to_logit(df[TARGET].values))
    model.fit(df, df[TARGET].values)

    # feature names out of the fitted preprocessor (for SHAP labels)
    fitted = model.regressor_
    feat_names = fitted.named_steps["prep"].get_feature_names_out().tolist()

    # ---- build metadata for the app form ---------------------------------- #
    def explode_choices(col, top=60):
        if col not in df.columns:
            return []
        return (df[col].explode().value_counts().head(top).index.astype(str).tolist())

    numeric_defaults = {c: float(np.nanmedian(pd.to_numeric(df.get(c, pd.Series([0])), errors="coerce")))
                        for c in BASE_NUMERICAL}
    metadata = {
        "metrics": metrics,
        "feature_names": feat_names,
        "genre_choices": explode_choices("genres"),
        "category_choices": explode_choices("categories"),
        "developer_choices": explode_choices("developers", top=300),
        "publisher_choices": explode_choices("publishers", top=300),
        "audio_lang_choices": explode_choices("full_audio_languages", top=60),
        "lang_group_choices": sorted(df["lang_group"].dropna().astype(str).unique().tolist())
            if "lang_group" in df.columns else ["1 язык", "2-5", "6-10", "11-20", "20+"],
        "numeric_defaults": numeric_defaults,
        "global_mean_rating": float(df[TARGET].mean()),
        "content_flags": [c for c in BASE_NUMERICAL if c.startswith("has_")],
    }

    # background sample for SHAP (raw rows -> the model transforms them)
    bg = df.sample(min(200, len(df)), random_state=42).reset_index(drop=True)

    joblib.dump(model, os.path.join(args.out, "model.pkl"))
    joblib.dump(metadata, os.path.join(args.out, "metadata.pkl"))
    joblib.dump(bg, os.path.join(args.out, "background.pkl"))

    # cleaned data for EDA (drop price as requested)
    eda = df.copy()
    if "price" in eda.columns:
        eda = eda.drop(columns=["price"])
    # store list columns as strings so parquet/csv is happy
    for c in LIST_COLS:
        if c in eda.columns:
            eda[c] = eda[c].apply(lambda v: v if isinstance(v, list) else parse_list(v))
            eda[c] = eda[c].apply(lambda v: "|".join(map(str, v)))
    eda.to_parquet(os.path.join(args.out, "eda_data.parquet"), index=False)

    print(f"\nSaved model + metadata + eda_data to ./{args.out}/")
    print(f"  background SHAP sample: {len(bg)} rows")


if __name__ == "__main__":
    main()
