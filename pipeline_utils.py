"""
Общий код конвейера (pipeline) для модели прогнозирования рейтинга в Steam.

Этот модуль переносит логику генерации признаков (feature engineering) 
из Jupyter-ноутбука в единый автономный трансформер `scikit-learn` (`SteamFeatureEngineer`)
и содержит вспомогательные функции для сборки полного сквозного конвейера.
Размещение этих пользовательских классов здесь (а не внутри скрипта обучения) позволяет приложению Streamlit загружать
сохраненную модель (десериализовать её), так как для процесса десериализации (unpickling) необходимо, чтобы определения классов были доступны для импорта.

Итоговый развернутый объект представляет собой:

TransformedTargetRegressor(
regressor = Pipeline([fe, preprocessor, LassoCV]),
func      = logit(y/100),
inverse   = expit(.) * 100
)

Таким образом, вызов метода `.predict(raw_df)` принимает строку данных в исходном формате (с колонками, указанными в начале ноутбука)
и возвращает прогнозируемую долю положительных отзывов, уже преобразованную обратно в шкалу от 0 до 100.
"""

import ast
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# --------------------------------------------------------------------------- #
#  Column definitions (mirror the notebook)                                    #
# --------------------------------------------------------------------------- #
TARGET = "pct_pos_total"
TEXT_FEATURE = "text_combined"

LIST_COLS = ["developers", "publishers", "categories", "genres", "full_audio_languages"]

CATEGORICAL_FEATURES = [
    "lang_group", "mac", "linux",
    "release_month", "release_dayofweek", "release_quarter", "season",
]

BASE_NUMERICAL = [
    "lang_count", "os_count", "audio_languages_count",
    "has_violence_gore", "has_sexual_content", "has_drugs_alcohol",
    "has_strong_language", "has_mature_themes", "has_no_information",
    "has_ru", "has_en", "weekend",
]

SEASON_MAP = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall",
}

_EPS = 1e-3


def to_logit(y):
    """Target transform: logit(y/100). Defined here so the pickle stays importable."""
    from scipy.special import logit
    return logit(np.clip(np.asarray(y, dtype=float) / 100.0, _EPS, 1 - _EPS))


def from_logit(z):
    """Inverse target transform: expit(z) * 100."""
    from scipy.special import expit
    return expit(np.asarray(z, dtype=float)) * 100.0


CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS.union({
    "game", "games", "play", "player", "will", "your",
    "build", "new", "world", "experience", "features",
    "including", "use", "using", "get", "make",
}))


# --------------------------------------------------------------------------- #
#  Helpers                                                                      #
# --------------------------------------------------------------------------- #
def parse_list(x):
    """Turn the various stored representations of a list column into a python list."""
    if isinstance(x, list):
        return [str(v) for v in x]
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return [str(i) for i in v]
            return [str(v)]
        except Exception:
            # fall back to comma separated
            return [p.strip() for p in s.split(",") if p.strip()]
    return [str(x)]


def _multilabel_binarize(lists, classes, frequent_set):
    """Manual multi-label binariser that is robust to unseen labels (-> 'other')."""
    idx = {c: i for i, c in enumerate(classes)}
    out = np.zeros((len(lists), len(classes)), dtype=float)
    for r, lst in enumerate(lists):
        for v in lst:
            key = v if v in frequent_set else "other"
            j = idx.get(key)
            if j is not None:
                out[r, j] = 1.0
    return out


# --------------------------------------------------------------------------- #
#  Feature engineer                                                             #
# --------------------------------------------------------------------------- #
class SteamFeatureEngineer(BaseEstimator, TransformerMixin):
    """Reproduces every manual preprocessing step from the notebook.

    fit(X, y) learns:
      * frequent genre / category sets (min_count=50) and their binary columns
      * frequent developer / publisher sets (min_count=5) and a smoothed
        target-encoding map (category_encoders-compatible smoothing).
    transform(X) emits a numeric/categorical/text feature frame ready for the
    downstream ColumnTransformer.

    Note: when wrapped in TransformedTargetRegressor the `y` seen here is the
    logit-transformed target, so the developer/publisher encoding lives in logit
    space.  Because the downstream StandardScaler normalises it and the transform
    is monotonic, predictions are equivalent to encoding on the raw target.
    """

    def __init__(self, genre_min_count=50, cat_min_count=50,
                 devpub_min_count=5, te_smoothing=10.0, te_min_samples_leaf=1):
        self.genre_min_count = genre_min_count
        self.cat_min_count = cat_min_count
        self.devpub_min_count = devpub_min_count
        self.te_smoothing = te_smoothing
        self.te_min_samples_leaf = te_min_samples_leaf

    # ---- engineering shared by fit & transform ---------------------------- #
    def _engineer(self, X):
        df = X.copy()

        # ensure every column we may touch exists, with sane defaults
        for c in LIST_COLS:
            if c not in df.columns:
                df[c] = [[] for _ in range(len(df))]
            df[c] = df[c].apply(parse_list)

        for c in ["about_the_game", "short_description"]:
            if c not in df.columns:
                df[c] = ""

        for c in BASE_NUMERICAL:
            if c not in df.columns and c not in ("audio_languages_count", "weekend"):
                df[c] = 0
        for c in ["lang_group"]:
            if c not in df.columns:
                df[c] = "unknown"
        for c in ["mac", "linux"]:
            if c not in df.columns:
                df[c] = 0

        # derived list features
        df["developer_first"] = df["developers"].apply(lambda x: x[0] if len(x) else "unknown")
        df["publisher_first"] = df["publishers"].apply(lambda x: x[0] if len(x) else "unknown")
        df["audio_languages_count"] = df["full_audio_languages"].apply(len)

        # date features
        if "release_date" not in df.columns:
            df["release_date"] = pd.NaT
        rd = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_month"] = rd.dt.month.fillna(1).astype(int)
        df["release_dayofweek"] = rd.dt.dayofweek.fillna(0).astype(int)
        df["release_quarter"] = rd.dt.quarter.fillna(1).astype(int)
        df["weekend"] = df["release_dayofweek"].isin([5, 6]).astype(int)
        df["season"] = df["release_month"].map(SEASON_MAP).fillna("winter")

        # text
        df[TEXT_FEATURE] = (
            df["about_the_game"].fillna("").astype(str) + " "
            + df["short_description"].fillna("").astype(str)
        )

        # make sure numerics are numeric
        for c in BASE_NUMERICAL:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        return df

    # ---- smoothed target encoding (category_encoders compatible) ---------- #
    def _fit_target_encoding(self, series, y):
        prior = float(np.mean(y))
        tmp = pd.DataFrame({"cat": series.values, "y": np.asarray(y)})
        stats = tmp.groupby("cat")["y"].agg(["count", "mean"])
        smoove = 1.0 / (1.0 + np.exp(-(stats["count"] - self.te_min_samples_leaf) / self.te_smoothing))
        enc = prior * (1.0 - smoove) + stats["mean"] * smoove
        return enc.to_dict(), prior

    # ---- fit -------------------------------------------------------------- #
    def fit(self, X, y):
        df = self._engineer(X)
        y = np.asarray(y, dtype=float)

        # genres / categories: frequent sets + binary column layout
        self.frequent_, self.classes_, self.binary_cols_ = {}, {}, {}
        for col, prefix, min_count in [
            ("genres", "genres", self.genre_min_count),
            ("categories", "categories", self.cat_min_count),
        ]:
            counts = df[col].explode().value_counts()
            frequent = set(str(x) for x in counts[counts >= min_count].index)
            classes = sorted(frequent) + ["other"]
            self.frequent_[col] = frequent
            self.classes_[col] = classes
            self.binary_cols_[col] = [f"{prefix}_{c}" for c in classes]

        # developer / publisher: rare grouping + smoothed target encoding
        self.devpub_frequent_, self.te_maps_, self.te_prior_ = {}, {}, {}
        for col in ["developer_first", "publisher_first"]:
            vc = df[col].value_counts()
            frequent = set(str(x) for x in vc[vc >= self.devpub_min_count].index)
            self.devpub_frequent_[col] = frequent
            grouped = df[col].where(df[col].isin(frequent), "other")
            enc_map, prior = self._fit_target_encoding(grouped, y)
            self.te_maps_[col] = enc_map
            self.te_prior_[col] = prior

        # final feature lists (dedup, preserve order) -> mirrors notebook
        genre_cols = self.binary_cols_["genres"]
        cat_cols = self.binary_cols_["categories"]
        self.numerical_features_ = list(dict.fromkeys(
            BASE_NUMERICAL + ["developer_first", "publisher_first"] + genre_cols + cat_cols
        ))
        self.categorical_features_ = list(CATEGORICAL_FEATURES)
        self.text_feature_ = TEXT_FEATURE
        self.output_columns_ = (
            self.numerical_features_ + self.categorical_features_ + [self.text_feature_]
        )
        return self

    # ---- transform -------------------------------------------------------- #
    def transform(self, X):
        df = self._engineer(X)

        # multi-label binary columns
        for col, prefix in [("genres", "genres"), ("categories", "categories")]:
            mat = _multilabel_binarize(
                df[col].tolist(), self.classes_[col], self.frequent_[col]
            )
            bin_df = pd.DataFrame(mat, columns=self.binary_cols_[col], index=df.index)
            df = pd.concat([df, bin_df], axis=1)

        # developer / publisher target encoding
        for col in ["developer_first", "publisher_first"]:
            frequent = self.devpub_frequent_[col]
            grouped = df[col].where(df[col].isin(frequent), "other")
            prior = self.te_prior_[col]
            df[col] = grouped.map(self.te_maps_[col]).fillna(prior).astype(float)

        # ensure categorical dtype is string-friendly for OneHotEncoder
        for c in self.categorical_features_:
            if c not in df.columns:
                df[c] = "unknown"

        out = df.reindex(columns=self.output_columns_)
        # fill any gaps
        for c in self.numerical_features_:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        out[self.text_feature_] = out[self.text_feature_].fillna("").astype(str)
        return out


# --------------------------------------------------------------------------- #
#  Pipeline builder                                                             #
# --------------------------------------------------------------------------- #
def build_preprocessor(numerical_features, categorical_features, text_feature):
    numerical_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("fs", SelectKBest(score_func=f_regression, k="all")),
    ])
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])
    text_transformer = Pipeline(steps=[
        ("vectorizer", TfidfVectorizer(
            max_features=200, min_df=5, max_df=0.95,
            ngram_range=(1, 2), sublinear_tf=True,
            stop_words=CUSTOM_STOP_WORDS,
        )),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    return ColumnTransformer(transformers=[
        ("numerical", numerical_transformer, numerical_features),
        ("categorical", categorical_transformer, categorical_features),
        ("text", text_transformer, text_feature),   # str -> 1-D series for tfidf
    ], remainder="drop")


def clean_feature_name(name: str) -> str:
    """Readable label for a ColumnTransformer output feature."""
    for p in ("numerical__", "categorical__", "text__"):
        if name.startswith(p):
            name = name[len(p):]
    name = name.replace("genres_", "Genre: ").replace("categories_", "Category: ")
    if name == "developer_first":
        name = "Developer reputation"
    elif name == "publisher_first":
        name = "Publisher reputation"
    return name.replace("_", " ")
