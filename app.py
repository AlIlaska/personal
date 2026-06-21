"""
app.py
======
Интерфейс Streamlit для оценки игр в Steam.

Три вкладки:

1. Одиночное прогнозирование - заполнить форму, получить прогнозируемый процент положительных отзывов.

2. Пакетное прогнозирование - загрузить CSV-файл, оцените каждую строку, скачайте результаты.

3. Панель мониторинга EDA - изучить, как жанр, время, цена и другие признаки связаны с рейтингами.

Модель загружается из файла `game_rating_model.pkl` (создается скриптом train_model.py).
Если этот файл отсутствует, вы можете загрузить pickle-файл или обучить модель в приложении из CSV-файла.

Запуск:  streamlit run app.py
"""

import io
import os
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline import (
    GameRatingPipeline,
    TARGET,
    engineer_raw_features,
)

st.set_page_config(page_title="Steam Rating Predictor", page_icon="🎮", layout="wide")

MODEL_PATH = "game_rating_model.pkl"


# --------------------------------------------------------------------------- #
# Pагрузка/обучение моделей
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_model_from_path(path: str) -> GameRatingPipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Тренируем модель 🏋️‍♂️ …")
def train_model_from_df(df: pd.DataFrame, model_type: str) -> GameRatingPipeline:
    return GameRatingPipeline(model_type=model_type).fit(df, verbose=False)


@st.cache_data(show_spinner=False)
def read_csv_cached(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def get_model():
    """Создание модели на основе данных из сессии, с диска, загруженных файлов или данных, полученных в процессе обучения в приложении."""
    if "model" in st.session_state:
        return st.session_state["model"]
    if os.path.exists(MODEL_PATH):
        model = load_model_from_path(MODEL_PATH)
        st.session_state["model"] = model
        return model
    return None


# --------------------------------------------------------------------------- #
# Боковая панель: статус модели и источники данных
# --------------------------------------------------------------------------- #
st.sidebar.title("🎮 Модель")
model = get_model()

if model is None:
    st.sidebar.warning("Обученная модель не загружена.")
    src = st.sidebar.radio("Получить модель:", ["Загрузить файл .pkl", "Обучить модель из CSV"])

    if src == "Загрузить файл .pkl":
        up = st.sidebar.file_uploader("Загрузить game_rating_model.pkl", type=["pkl"])
        if up is not None:
            st.session_state["model"] = pickle.load(io.BytesIO(up.read()))
            st.rerun()
    else:
        train_csv = st.sidebar.file_uploader("Обучить модель из CSV (Final_clean_df.csv)", type=["csv"])
        mtype = st.sidebar.selectbox("Тип модели", ["lasso", "ridge", "elasticnet"])
        if train_csv is not None and st.sidebar.button("Train now"):
            df_train = read_csv_cached(train_csv.getvalue())
            st.session_state["model"] = train_model_from_df(df_train, mtype)
            st.session_state["train_df_bytes"] = train_csv.getvalue()
            st.rerun()
    st.title("Steam Game Rating Predictor")
    st.info("Загрузите или обучите модель из боковой панели, чтобы начать ❗ "
            "Самый быстрый способ - запустить по команде`python train_model.py --data Final_clean_df.csv` "
            "первый раз, а затем запустить приложение. Оно автоматически подхватит файл pickle.")
    st.stop()

# model is available
m = model.metrics_.get("test", {})
st.sidebar.success("Модель загружена ✓")
st.sidebar.metric("Test RMSE", f"{m.get('rmse', float('nan')):.2f}")
st.sidebar.metric("Test MAE", f"{m.get('mae', float('nan')):.2f}")
st.sidebar.metric("Test R²", f"{m.get('r2', float('nan')):.3f}")
st.sidebar.caption(
    f"Model: **{model.model_type}** · {model.n_train_:,} train / {model.n_test_:,} test rows"
)
if model.metrics_.get("test_date_range"):
    a, b = model.metrics_["test_date_range"]
    st.sidebar.caption(f"Период тестирования: {a} → {b}")
if st.sidebar.button("Сбросить модель"):
    st.session_state.pop("model", None)
    st.cache_resource.clear()
    st.rerun()


# --------------------------------------------------------------------------- #
# Вкладки
# --------------------------------------------------------------------------- #
tab_single, tab_batch, tab_eda = st.tabs(
    ["🎯 Одиночное предсказание", "📦 Группа (CSV)", "📊 Разведочный анализ данных"]
)


# ===================== Первая страница - Единичное предсказание =========================== #
with tab_single:
    st.header("Спрогнозируйте рейтинг одной игры 📶")
    st.caption("Укажите метаданные, которые вам известны **до** релиза. "
               "Модель возвращает ожидаемую долю положительных отзывов (0–100%).")

    genre_opts = model.genre_options()
    cat_opts = model.category_options()
    lang_group_opts = model.categorical_values_.get(
        "lang_group", ["en_only", "ru_en", "multi", "other"])

    c1, c2, c3 = st.columns(3)
    with c1:
        name = st.text_input("Имя", "Новая игра")
        release_date = st.date_input("Дата релиза", pd.Timestamp("2025-06-15"))
        price = st.number_input("Цена ($)", min_value=0.0, value=19.99, step=1.0)
        lang_group = st.selectbox("Группа языков", lang_group_opts)
        lang_count = st.number_input("Количество поддерживаемых языков", 1, 40, 5)
    with c2:
        genres = st.multiselect("Жанры", genre_opts, default=genre_opts[:1])
        categories = st.multiselect("Категории", cat_opts, default=cat_opts[:1])
        audio_langs = st.number_input("Количество аудио-языков", 0, 40, 1)
        developer = st.text_input("Раззработчик", "unknown")
        publisher = st.text_input("Издатель", "unknown")
    with c3:
        st.markdown("**Платформы**")
        win = st.checkbox("Windows", True, disabled=True)
        mac = st.checkbox("macOS", False)
        linux = st.checkbox("Linux", False)
        st.markdown("**Языковые индикаторы**")
        has_en = st.checkbox("Поддерживает английский язык (English)", True)
        has_ru = st.checkbox("Поддерживает русский язык (Russian)", False)

    st.markdown("**Особые индикаторы контента**")
    f1, f2, f3 = st.columns(3)
    with f1:
        has_violence_gore = st.checkbox("Насилие / кровь")
        has_sexual_content = st.checkbox("Сексуальный контент")
    with f2:
        has_drugs_alcohol = st.checkbox("Наркотики* / алкоголь")
        has_strong_language = st.checkbox("Нецензурная лексика")
    with f3:
        has_mature_themes = st.checkbox("Темы для взрослых")
        has_no_information = st.checkbox("Информация о содержании отсутствует")

    about = st.text_area(
        "Об игре (текст описания)",
        "Эпическое сюжетное приключение с глубокой боевой системой и исследованием мира.",
        height=120,
    )

    if st.button("Предсказать", type="primary"):
        os_count = 1 + int(mac) + int(linux)
        record = pd.DataFrame([{
            "name": name,
            "release_date": str(release_date),
            "price": price,
            "year": pd.Timestamp(release_date).year,
            "genres": repr(genres),
            "categories": repr(categories),
            "developers": repr([developer] if developer else []),
            "publishers": repr([publisher] if publisher else []),
            "full_audio_languages": repr(["lang"] * int(audio_langs)),
            "about_the_game": about,
            "mac": int(mac), "linux": int(linux),
            "os_count": os_count, "lang_count": int(lang_count),
            "lang_group": lang_group,
            "has_violence_gore": int(has_violence_gore),
            "has_sexual_content": int(has_sexual_content),
            "has_drugs_alcohol": int(has_drugs_alcohol),
            "has_strong_language": int(has_strong_language),
            "has_mature_themes": int(has_mature_themes),
            "has_no_information": int(has_no_information),
            "has_ru": int(has_ru), "has_en": int(has_en),
        }])
        pred = float(model.predict(record)[0])

        st.markdown("### Результат")
        rc1, rc2 = st.columns([1, 2])
        with rc1:
            st.metric("Прогнозируемые положительные отзывы", f"{pred:.1f}%")
        with rc2:
            band = ("Подавляющее большинство/Очень положительные" if pred >= 80
                    else "В основном положительные" if pred >= 70
                    else "Смешанные" if pred >= 40
                    else "В основном негативные")
            st.progress(min(pred / 100, 1.0))
            st.caption(f"Ожидаемый диапазон пользовательского мнения в Steam: **{band}** "
                       f"(±{m.get('rmse', 12):.0f} pts typical error)")
        
    st.caption("_*Незаконное потребление наркотических средств, психотропных веществ, их аналогов причиняет вред здоровью, их незаконный оборот запрещен и влечет установленную законодательством ответственность._")


# =========== Второй лист - Несколько игр через CSV 
with tab_batch:
    st.header("Результаты игр в формате CSV")
    st.caption("Загрузите таблицу в том же формате, что и обучающие данные ⚠️"
               "(Столбец `pct_pos_total` является необязательным, но если он присутствует, отобразиться ошибка).")
    up = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")

    if up is not None:
        df_in = read_csv_cached(up.getvalue())
        st.write(f"Загружено **{len(df_in):,}** объектов.")
        with st.spinner("Прогнозирование …"):
            preds = model.predict(df_in)
        out = df_in.copy()
        out["predicted_pct_pos"] = np.round(preds, 2)

        cols_preview = [c for c in ["name", "release_date", "genres",
                                    "predicted_pct_pos", TARGET] if c in out.columns]
        st.dataframe(out[cols_preview].head(200), use_container_width=True)

        if TARGET in df_in.columns:
            mask = df_in[TARGET].notna()
            if mask.any():
                err = (out.loc[mask, "predicted_pct_pos"] - df_in.loc[mask, TARGET])
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("MAE", f"{err.abs().mean():.2f}")
                mc2.metric("RMSE", f"{np.sqrt((err**2).mean()):.2f}")
                mc3.metric("Rows scored", f"{int(mask.sum()):,}")
                fig = px.scatter(
                    x=df_in.loc[mask, TARGET], y=out.loc[mask, "predicted_pct_pos"],
                    labels={"x": "Факт %", "y": "Прогноз %"},
                    title="Прогноз vs факт", opacity=0.5,
                )
                fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                              line=dict(dash="dash"))
                st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "⬇️ Загрузка результатов",
            out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv", mime="text/csv",
        )


# ========================= TAB 3 — EDA DASHBOARD =========================== #
with tab_eda:
    st.header("Исследовательская панель🕵️")
    st.caption("Загрузите набор данных (в том же формате, что и для обучения), чтобы изучить, как влияет жанр, "
               "время, цена и контент связан с рейтингом.")

    eda_up = st.file_uploader("Загрузить набор данных CSV", type=["csv"], key="eda_csv")
    raw = None
    if eda_up is not None:
        raw = read_csv_cached(eda_up.getvalue())
    elif "train_df_bytes" in st.session_state:
        raw = read_csv_cached(st.session_state["train_df_bytes"])
        st.info("Используйте набор данных, на котором вы обучали модель / Загрузите другой CSV-файл для замены.")

    if raw is None:
        st.warning("Загрузите CSV-файл для заполнения панели мониторинга.")
        st.stop()

    if TARGET not in raw.columns:
        st.error(f"Целевая переменная `{TARGET}`- не найдена, на панели управления необходим столбец с рейтингом.")
        st.stop()

    # подготовка аккуратной рамочки
    @st.cache_data(show_spinner=False)
    def prepare_eda(df_bytes: bytes) -> pd.DataFrame:
        df = pd.read_csv(io.BytesIO(df_bytes))
        df = engineer_raw_features(df)
        df = df.dropna(subset=[TARGET, "release_date"])
        df = df[df["year"] > 2008]
        return df

    df = prepare_eda(eda_up.getvalue() if eda_up is not None
                     else st.session_state["train_df_bytes"])

    # explode genres for genre-level analysis
    g = df[["year", "season", "price", TARGET, "genres"]].explode("genres")
    g = g[g["genres"].notna() & (g["genres"] != "")]

    # --- фильтры ---
    yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
    fc1, fc2 = st.columns([2, 3])
    with fc1:
        yr_range = st.slider("Диапазон годов выпуска", yr_min, yr_max, (yr_min, yr_max))
    top_genres = g["genres"].value_counts().head(12).index.tolist()
    with fc2:
        sel_genres = st.multiselect("Genres to include", top_genres,
                                    default=top_genres[:6])

    dff = df[(df["year"] >= yr_range[0]) & (df["year"] <= yr_range[1])]
    gff = g[(g["year"] >= yr_range[0]) & (g["year"] <= yr_range[1])
            & (g["genres"].isin(sel_genres))]

    # основные показатели
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Games", f"{len(dff):,}")
    k2.metric("Avg rating", f"{dff[TARGET].mean():.1f}%")
    k3.metric("Median rating", f"{dff[TARGET].median():.1f}%")
    k4.metric("Free games", f"{(dff['price'].fillna(0) == 0).mean()*100:.0f}%")

    st.divider()

    # 1 - Влияние жанра с течением времени (общий взгляд)
    st.subheader("Как жанр влияет на рейтинг с течением времени⏳")
    cc1, cc2 = st.columns(2)
    with cc1:
        gy = (gff.groupby(["year", "genres"])[TARGET].mean().reset_index())
        fig = px.line(gy, x="year", y=TARGET, color="genres", markers=True,
                      labels={TARGET: "Avg positive %"},
                      title="Средний рейтинг по жанрам за год.")
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        heat = gff.groupby(["genres", "year"])[TARGET].mean().reset_index()
        heat_p = heat.pivot(index="genres", columns="year", values=TARGET)
        fig = px.imshow(heat_p, aspect="auto", color_continuous_scale="RdYlGn",
                        labels=dict(color="Avg %"),
                        title="Тепловая карта рейтинга по жанрам и годам")
        st.plotly_chart(fig, use_container_width=True)

    # 2 - жанровый рейтинг + распределение
    cc1, cc2 = st.columns(2)
    with cc1:
        rank = (gff.groupby("genres")[TARGET]
                .agg(["mean", "count"]).reset_index()
                .sort_values("mean", ascending=True))
        fig = px.bar(rank, x="mean", y="genres", orientation="h",
                     color="mean", color_continuous_scale="RdYlGn",
                     hover_data=["count"], labels={"mean": "Avg positive %"},
                     title="Средний рейтинг по жанрам (за выбранный период)")
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        fig = px.box(gff, x="genres", y=TARGET, color="genres",
                     labels={TARGET: "Positive %"},
                     title="Распределение рейтингов по жанрам")
        fig.update_layout(showlegend=False, xaxis_tickangle=-40)
        st.plotly_chart(fig, use_container_width=True)

    # 3- общие тенденции
    st.subheader("Общие тенденции 📊")
    cc1, cc2 = st.columns(2)
    with cc1:
        trend = dff.groupby("year")[TARGET].agg(["mean", "count"]).reset_index()
        fig = px.line(trend, x="year", y="mean", markers=True,
                      labels={"mean": "Avg positive %"},
                      title="Средний рейтинг по всем играм за год.")
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        rel = dff.groupby("year").size().reset_index(name="releases")
        fig = px.bar(rel, x="year", y="releases", title="Выпуски/год")
        st.plotly_chart(fig, use_container_width=True)

    # 4 - Цена и сезонность
    cc1, cc2 = st.columns(2)
    with cc1:
        tmp = dff.copy()
        tmp["price_bucket"] = pd.cut(
            tmp["price"].fillna(0),
            bins=[-0.01, 0, 5, 10, 20, 40, 1e9],
            labels=["Free", "$0–5", "$5–10", "$10–20", "$20–40", "$40+"])
        pb = tmp.groupby("price_bucket", observed=True)[TARGET].mean().reset_index()
        fig = px.bar(pb, x="price_bucket", y=TARGET, color=TARGET,
                     color_continuous_scale="RdYlGn",
                     labels={TARGET: "Avg positive %"},
                     title="Рейтинг по ценовым категориям")
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        if "season" in dff.columns:
            order = ["winter", "spring", "summer", "fall"]
            se = (dff.groupby("season")[TARGET].mean()
                  .reindex(order).reset_index())
            fig = px.bar(se, x="season", y=TARGET, color=TARGET,
                         color_continuous_scale="RdYlGn",
                         labels={TARGET: "Avg positive %"},
                         title="Рейтинг по сезонам выпуска")
            st.plotly_chart(fig, use_container_width=True)

    # 5 - распределеине рейтинга
    st.subheader("Rating distribution")
    fig = px.histogram(dff, x=TARGET, nbins=40,
                       labels={TARGET: "Positive %"},
                       title="Распределение доли положительных отзывов")
    st.plotly_chart(fig, use_container_width=True)
