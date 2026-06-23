"""Explanation helpers for the linear (Lasso) Steam rating model.

For a linear model, SHAP values are exactly coef * (x - E[x]) in the model's
(logit) space.  We compute them with shap.LinearExplainer (falling back to the
closed form if shap is unavailable) and rescale to approximate percentage-point
effects using the logit derivative at the prediction, exactly as in the notebook.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit

import pipeline_utils as pu


def _dense(m):
    return m.toarray() if sparse.issparse(m) else np.asarray(m)


def unwrap(model):
    """Return (feature_engineer, preprocessor, linear_model) from the saved object."""
    inner = getattr(model, "regressor_", model)
    steps = inner.named_steps
    return steps["fe"], steps["prep"], steps["model"]


def transformed(model, X):
    fe, prep, _ = unwrap(model)
    return _dense(prep.transform(fe.transform(X)))


def feature_names(model):
    _, prep, _ = unwrap(model)
    return [pu.clean_feature_name(n) for n in prep.get_feature_names_out()]


def feature_weights(model, top=20):
    """Global Lasso coefficients (the model is linear) -> tidy DataFrame."""
    _, _, lin = unwrap(model)
    df = pd.DataFrame({"feature": feature_names(model), "weight": lin.coef_})
    df = df[df["weight"] != 0].copy()
    df["abs"] = df["weight"].abs()
    return df.sort_values("abs", ascending=False).head(top).drop(columns="abs")


def local_explanation(model, background_df, X_row):
    """Вклад SHAP для одной записи, масштабированный до ~процентных пунктов.
    Возвращает словарь, содержащий: DataFrame вкладов (признак, SHAP), базовое значение (%),
    прогноз (%) и исходный вектор SHAP для необязательного сводного графика.
    """
    fe, prep, lin = unwrap(model)
    names = feature_names(model)
    Xb = _dense(prep.transform(fe.transform(background_df)))
    Xr = _dense(prep.transform(fe.transform(X_row)))

    try:
        import shap
        explainer = shap.LinearExplainer(lin, Xb)
        sv = explainer.shap_values(Xr)
        base_logit = float(np.atleast_1d(explainer.expected_value)[0])
    except Exception:
        # closed form interventional SHAP for a linear model
        mean = Xb.mean(axis=0)
        sv = (Xr - mean) * lin.coef_
        base_logit = float(lin.intercept_ + (lin.coef_ * mean).sum())

    pred_logit = float(lin.predict(Xr)[0])
    # logit -> percent scaling (derivative of expit*100 at the prediction)
    scale = expit(pred_logit) * (1 - expit(pred_logit)) * 100.0
    contrib = sv[0] * scale

    df = pd.DataFrame({"feature": names, "shap": contrib})
    df = df[df["shap"].abs() > 1e-6]
    return {
        "contributions": df,
        "base_pct": float(expit(base_logit) * 100.0),
        "prediction_pct": float(expit(pred_logit) * 100.0),
    }


def global_shap_importance(model, background_df, top=15):
    """Mean |SHAP| over the background sample -> tidy DataFrame."""
    fe, prep, lin = unwrap(model)
    names = feature_names(model)
    Xb = _dense(prep.transform(fe.transform(background_df)))
    try:
        import shap
        sv = shap.LinearExplainer(lin, Xb).shap_values(Xb)
    except Exception:
        mean = Xb.mean(axis=0)
        sv = (Xb - mean) * lin.coef_
    imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    return imp.sort_values("mean_abs_shap", ascending=False).head(top)
