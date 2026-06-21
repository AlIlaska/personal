"""
pipeline.py
===========
Автономный, сериализуемый pipeline, воспроизводящий модель рейтинга игр Steam
из ноутбука (прогнозирующий `pct_pos_total`, процент положительных отзывов, на основе
метаданных до релиза).

Все необходимое для модели находится внутри объекта `GameRatingPipeline`:

* детерминированная разработка признаков (части даты, разбор списков, подсчет аудио)

* обученные «ручные» трансформеры (группировка редких категорий, MultiLabelBinarizer
для жанров и категорий, TargetEncoder для разработчика/издателя)

* обученный pipeline sklearn (ColumnTransformer + Lasso/Ridge/ElasticNet)

* метрики на отложенной выборке + метаданные

Один и тот же класс импортируется как в `train_model.py` (для обучения и типо "сериализации"), так и в
`app.py` (для загрузки и прогнозирования), поэтому десериализация всегда находит определение класса.

Примечание по сравнению с оригинальным блокнотом: в блокноте бинаризировались только категории 
(ячейка MLB выполнялась после цикла жанров/категорий, поэтому `genre_cols` оставалась пустой). 
Этот модуль исправляет это и бинаризует как жанры, так и категории, что является предполагаемым поведением 
и на чем основан анализ жанров методом разведочного анализа.
"""

from __future__ import annotations

import ast
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MultiLabelBinarizer,
    OneHotEncoder,
    StandardScaler,
)

import category_encoders as ce


# --------------------------------------------------------------------------- #
# Определения столбцов (зеркалььно ноутбуку)
# --------------------------------------------------------------------------- #
LIST_COLS = ["developers", "publishers", "categories", "genres", "full_audio_languages"]

CATEGORICAL_FEATURES = [
    "lang_group", "mac", "linux",
    "release_month", "release_dayofweek", "release_quarter", "season",
]

# базовые числовые характеристики до добавления столбцов с кодировкой MLB/Target Encoding
NUMERIC_BASE = [
    "lang_count", "os_count", "audio_languages_count",
    "has_violence_gore", "has_sexual_content", "has_drugs_alcohol",
    "has_strong_language", "has_mature_themes", "has_no_information",
    "has_ru", "has_en", "weekend",
]

TEXT_FEATURE = "about_the_game"
TARGET = "pct_pos_total"

SEASON_MAP = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall",
}

EXTRA_STOP_WORDS = {
    "game", "games", "play", "player", "will", "your",
    "build", "new", "world", "experience", "features",
    "including", "use", "using", "get", "make",
}

EPSILON = 1e-3  # сохраняет logit в диапазоне 0 / 100


# --------------------------------------------------------------------------- #
# feature engineering (подгонка не требуется)
# --------------------------------------------------------------------------- #
def _parse_list(value):
    """Turn a stringified list into a real list; pass real lists through."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except (ValueError, SyntaxError):
            # простая строка без запятых - отдельный элемент
            return [s]
    return []


def engineer_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Примените детерминированное, непостоянное feature engineering из блокнота.
    Безопасно вызывать и для одной записи, и для всей таблицы. Возвращает НОВЫЙ датафрейм.
    """
    df = df.copy()

    # 1. Парсим лист колонок
    for col in LIST_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_list)
        else:
            df[col] = [[] for _ in range(len(df))]

    # 2. Первый разработчик/издатель + количество аудио-языков
    df["developer_first"] = df["developers"].apply(lambda x: x[0] if x else "unknown")
    df["publisher_first"] = df["publishers"].apply(lambda x: x[0] if x else "unknown")
    df["audio_languages_count"] = df["full_audio_languages"].apply(len)

    # 3. date-derived features
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_month"] = df["release_date"].dt.month
    df["release_dayofweek"] = df["release_date"].dt.dayofweek
    df["release_quarter"] = df["release_date"].dt.quarter
    df["weekend"] = df["release_dayofweek"].isin([5, 6]).astype("Int64").astype(float)
    df["season"] = df["release_month"].map(SEASON_MAP)
    if "year" not in df.columns:
        df["year"] = df["release_date"].dt.year

    # 4. Текст (описание)
    df[TEXT_FEATURE] = df.get(TEXT_FEATURE, "").fillna("") if TEXT_FEATURE in df.columns \
        else ""

    # 5. Проверка, что все исходные числовые/категориальные данные присутствуют
    for col in NUMERIC_BASE:
        if col not in df.columns:
            df[col] = 0
    for col in ["lang_group", "mac", "linux"]:
        if col not in df.columns:
            df[col] = "unknown" if col == "lang_group" else 0

    return df


def _replace_rare_in_list(lst, frequent):
    """Replace list items not in `frequent` with 'other', dropping duplicates."""
    out, seen = [], set()
    for v in lst:
        token = v if v in frequent else "other"
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


# --------------------------------------------------------------------------- 
# Объект pipeline
# --------------------------------------------------------------------------
@dataclass
class GameRatingPipeline:
    model_type: str = "lasso"          # 'lasso' | 'ridge' | 'elasticnet'
    genre_min_count: int = 50          # rare threshold for genres
    cat_min_count: int = 50            # rare threshold for categories
    dev_pub_min_count: int = 5         # rare threshold for developer / publisher

    # ---- fitted artefacts (populated by .fit) ----
    genre_frequent_: set = field(default_factory=set)
    cat_frequent_: set = field(default_factory=set)
    dev_frequent_: set = field(default_factory=set)
    pub_frequent_: set = field(default_factory=set)
    genre_mlb_: Optional[MultiLabelBinarizer] = None
    cat_mlb_: Optional[MultiLabelBinarizer] = None
    target_encoder_: Optional[ce.TargetEncoder] = None
    sk_pipeline_: Optional[Pipeline] = None
    genre_cols_: list = field(default_factory=list)
    cat_cols_: list = field(default_factory=list)
    numeric_features_: list = field(default_factory=list)
    feature_cols_: list = field(default_factory=list)
    metrics_: dict = field(default_factory=dict)
    categorical_values_: dict = field(default_factory=dict)  # для выпадающих списков пользовательского интерфейса
    n_train_: int = 0
    n_test_: int = 0
    is_fitted_: bool = False

    # списки удобных параметров для пользовательского интерфейса
    def genre_options(self) -> list:
        opts = [c for c in (self.genre_mlb_.classes_ if self.genre_mlb_ else []) if c != "other"]
        return opts or ["Action", "Adventure", "RPG", "Strategy", "Casual",
                        "Indie", "Simulation", "Sports", "Racing", "Free to Play"]

    def category_options(self) -> list:
        opts = [c for c in (self.cat_mlb_.classes_ if self.cat_mlb_ else []) if c != "other"]
        return opts or ["Single-player", "Multi-player", "Steam Cloud",
                        "Steam Achievements", "In-App Purchases", "Co-op",
                        "Full controller support"]

    # -----------------------------------------------------------------
    # внутри - построение матрицы признаков, готовой к использованию в модели.

    def _build_feature_matrix(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        На основе уже обработанного с помощью engineer_raw_features() датафрейма применяется
        готовый к модели X. Если fit=True, обучаются наборы данных MLB/TargetEncoder/группировка редких значений; 
        в противном случае они применяются.
        """
        df = df.copy()

        # Тут жанры и категории проходят MultiLabelBinarizer
        if fit:
            g_counts = df["genres"].explode().value_counts()
            self.genre_frequent_ = set(g_counts[g_counts >= self.genre_min_count].index)
            c_counts = df["categories"].explode().value_counts()
            self.cat_frequent_ = set(c_counts[c_counts >= self.cat_min_count].index)

        g_lists = df["genres"].apply(lambda l: _replace_rare_in_list(l, self.genre_frequent_))
        c_lists = df["categories"].apply(lambda l: _replace_rare_in_list(l, self.cat_frequent_))

        if fit:
            self.genre_mlb_ = MultiLabelBinarizer().fit(g_lists)
            self.cat_mlb_ = MultiLabelBinarizer().fit(c_lists)
            self.genre_cols_ = [f"genres_{c}" for c in self.genre_mlb_.classes_]
            self.cat_cols_ = [f"categories_{c}" for c in self.cat_mlb_.classes_]

        g_dummies = pd.DataFrame(
            self.genre_mlb_.transform(g_lists),
            columns=self.genre_cols_, index=df.index,
        )
        c_dummies = pd.DataFrame(
            self.cat_mlb_.transform(c_lists),
            columns=self.cat_cols_, index=df.index,
        )
        df = pd.concat([df, g_dummies, c_dummies], axis=1)

        # разработчик/издатель: группировка редких + целевое кодирование 
        if fit:
            dev_counts = df["developer_first"].value_counts()
            self.dev_frequent_ = set(dev_counts[dev_counts >= self.dev_pub_min_count].index)
            pub_counts = df["publisher_first"].value_counts()
            self.pub_frequent_ = set(pub_counts[pub_counts >= self.dev_pub_min_count].index)

        df["developer_first"] = df["developer_first"].apply(
            lambda x: x if x in self.dev_frequent_ else "other")
        df["publisher_first"] = df["publisher_first"].apply(
            lambda x: x if x in self.pub_frequent_ else "other")

        if fit:
            y_logit = logit(np.clip(df[TARGET] / 100, EPSILON, 1 - EPSILON))
            self.target_encoder_ = ce.TargetEncoder(
                cols=["developer_first", "publisher_first"], smoothing=10)
            self.target_encoder_.fit(
                df[["developer_first", "publisher_first"]], y_logit)
            # assemble final feature list once
            self.numeric_features_ = list(dict.fromkeys(
                NUMERIC_BASE + ["developer_first", "publisher_first"]
                + self.genre_cols_ + self.cat_cols_))
            self.feature_cols_ = list(dict.fromkeys(
                self.numeric_features_ + CATEGORICAL_FEATURES + [TEXT_FEATURE]))

        df[["developer_first", "publisher_first"]] = self.target_encoder_.transform(
            df[["developer_first", "publisher_first"]])

        # обеспечить наличие всех ожидаемых столбцов (устойчивость к ранее не встречавшимся отдельным записям)
        for col in self.feature_cols_:
            if col not in df.columns:
                df[col] = 0
        return df[self.feature_cols_].copy()

    # ------------------------------------------------------------------ 
    def _make_sk_pipeline(self) -> Pipeline:
        stop_words = list(ENGLISH_STOP_WORDS.union(EXTRA_STOP_WORDS))

        numeric_tf = Pipeline([
            ("scaler", StandardScaler()),
            ("fs", SelectKBest(score_func=f_regression, k="all")),
        ])
        categorical_tf = Pipeline([
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ])
        text_tf = Pipeline([
            ("vectorizer", TfidfVectorizer(
                max_features=200, min_df=5, max_df=0.95,
                ngram_range=(1, 2), sublinear_tf=True, stop_words=stop_words)),
            ("scaler", StandardScaler(with_mean=False)),
        ])

        data_tf = ColumnTransformer([
            ("numerical", numeric_tf, self.numeric_features_),
            ("categorical", categorical_tf, CATEGORICAL_FEATURES),
            ("text", text_tf, TEXT_FEATURE),
        ], remainder="drop")

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        if self.model_type == "ridge":
            model = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=kf)
        elif self.model_type == "elasticnet":
            model = ElasticNetCV(
                alphas=np.logspace(-3, 2, 20),
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                cv=kf, max_iter=10000)
        else:  # lasso (best in the notebook)
            model = LassoCV(
                alphas=np.logspace(-3, 2, 20), cv=kf,
                max_iter=3000, n_jobs=-1, tol=1e-3)

        return Pipeline([("preprocessor", data_tf), ("model", model)])

    # ---------------------------------------------------------------
    @staticmethod
    def _metrics(y_true, y_pred) -> dict:
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    # ----------------------------------------------------------------
    def fit(self, df: pd.DataFrame, test_size: float = 0.2, verbose: bool = True):
        """
        Reproduce the notebook training:
          * deterministic feature engineering + filters (year>2008, price>=0)
          * time-based train/test split (earliest 80% train, latest 20% test)
          * fit all transformers + the chosen linear model on train
          * evaluate on the time-held-out test set (stored in .metrics_)
        The deployed artefact is the train-fitted pipeline (faithful to notebook).
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            df = engineer_raw_features(df)
            if "year" in df.columns:
                df = df[df["year"] > 2008]
            if "price" in df.columns:
                df = df[df["price"].fillna(0) >= 0]
            df = df.dropna(subset=[TARGET, "release_date"]).reset_index(drop=True)

            # time split
            df = df.sort_values("release_date").reset_index(drop=True)
            split = int(len(df) * (1 - test_size))
            train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()
            self.n_train_, self.n_test_ = len(train_df), len(test_df)

            # fit feature matrix on train, transform test
            X_train = self._build_feature_matrix(train_df, fit=True)
            X_test = self._build_feature_matrix(test_df, fit=False)
            y_train = train_df[TARGET].values
            y_test = test_df[TARGET].values
            y_train_logit = logit(np.clip(y_train / 100, EPSILON, 1 - EPSILON))

            # fit model
            self.sk_pipeline_ = self._make_sk_pipeline()
            self.sk_pipeline_.fit(X_train, y_train_logit)

            # metrics (inverse-transform predictions to 0-100)
            tr_pred = expit(self.sk_pipeline_.predict(X_train)) * 100
            te_pred = expit(self.sk_pipeline_.predict(X_test)) * 100
            self.metrics_ = {
                "train": self._metrics(y_train, tr_pred),
                "test": self._metrics(y_test, te_pred),
                "train_date_range": [str(train_df["release_date"].min().date()),
                                     str(train_df["release_date"].max().date())],
                "test_date_range": [str(test_df["release_date"].min().date()),
                                    str(test_df["release_date"].max().date())],
            }
            self.is_fitted_ = True

            # capture categorical option lists for the UI
            self.categorical_values_ = {}
            for col in ["lang_group", "season", "release_month",
                        "release_quarter", "release_dayofweek"]:
                if col in train_df.columns:
                    vals = sorted(train_df[col].dropna().unique().tolist())
                    self.categorical_values_[col] = vals

        if verbose:
            m = self.metrics_
            print(f"Trained '{self.model_type}'  "
                  f"(train={self.n_train_}, test={self.n_test_})")
            print(f"  Test  RMSE={m['test']['rmse']:.3f}  "
                  f"MAE={m['test']['mae']:.3f}  R2={m['test']['r2']:.3f}")
        return self

    # ------------------------------------------------------------------ #
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict pct_pos_total (0-100) for a record or table of raw inputs."""
        if not self.is_fitted_:
            raise RuntimeError("Pipeline is not fitted. Call .fit() or load a trained model.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eng = engineer_raw_features(df)
            X = self._build_feature_matrix(eng, fit=False)
            pred_logit = self.sk_pipeline_.predict(X)
        return np.clip(expit(pred_logit) * 100, 0, 100)
