"""
Streamlit app.
Run:
    streamlit run app.py

Expects ./artifacts/ produced by train_model.py:
    model.pkl, metadata.pkl, background.pkl, eda_data.parquet
"""
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import pipeline_utils as pu
import explain as ex

warnings.filterwarnings("ignore")

ART = "artifacts"
GREEN = "#2E9E5B"
RED = "#D1495B"
ACCENT = "#1B6CA8"
PIE_TEMPLATE = "plotly_white"

st.set_page_config(page_title="Прогнозирование рейтинга в Steam", page_icon="🎮", layout="wide")


# --------------------------------------------------------------------------- #
#  Loading                                                                      #
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(os.path.join(ART, "model.pkl"))
    meta = joblib.load(os.path.join(ART, "metadata.pkl"))
    bg = joblib.load(os.path.join(ART, "background.pkl"))
    return model, meta, bg


@st.cache_data(show_spinner=False)
def load_eda():
    path = os.path.join(ART, "eda_data.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    for c in pu.LIST_COLS:
        if c in df.columns:
            df[c + "_list"] = df[c].fillna("").apply(
                lambda s: [p for p in str(s).split("|") if p]
            )
    if "year" not in df.columns and "release_date" in df.columns:
        df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    return df

@st.cache_data(show_spinner=False)
def read_eda_csv(file_bytes):
    """Прочитает загруженный CSV-файл в исходном формате и приведет его к виду, соответствующему выводу функции load_eda()."""
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))
    if "price" in df.columns:                       # never used in the dashboard
        df = df.drop(columns=["price"])
    for c in pu.LIST_COLS:                           # parse "['Action','RPG']" -> list
        if c in df.columns:
            df[c + "_list"] = df[c].apply(pu.parse_list)
    if "year" not in df.columns and "release_date" in df.columns:
        df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    if pu.TARGET in df.columns:
        df[pu.TARGET] = pd.to_numeric(df[pu.TARGET], errors="coerce")
    return df


if not os.path.exists(os.path.join(ART, "model.pkl")):
    st.error(
        "Обученная модель не найдена. Выполните `python train_model.py --data clean_data_v1.csv` "
        "впервые, что бы создать ./artifacts/ folder."
    )
    st.stop()

model, META, BG = load_artifacts()


# --------------------------------------------------------------------------- #
#  Визуализация пояснения                                                     #
# --------------------------------------------------------------------------- #
def render_explanation(X_row):
    res = ex.local_explanation(model, BG, X_row)
    contrib = res["contributions"].sort_values("shap")

    top_green = contrib.tail(3).iloc[::-1]
    top_red = contrib.head(3)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🟢 Основные факторы, способствующие **повышению** рейтинга.")
        if top_green.empty:
            st.caption("Нет положительного влияния для этой записи.")
        for _, r in top_green.iterrows():
            st.markdown(f"**{r['feature']}**  ·  +{r['shap']:.2f} pts")
    with c2:
        st.markdown("##### 🔴 Основные факторы, способствующие **снижению** рейтинга.")
        if top_red.empty:
            st.caption("Нет негативного вляния для этой записи")
        for _, r in top_red.iterrows():
            st.markdown(f"**{r['feature']}**  ·  {r['shap']:.2f} pts")

    # combined green/red horizontal bar
    plot_df = pd.concat([top_red, top_green]).drop_duplicates("feature")
    plot_df = plot_df.sort_values("shap")
    fig = go.Figure(go.Bar(
        x=plot_df["shap"], y=plot_df["feature"], orientation="h",
        marker_color=[GREEN if v > 0 else RED for v in plot_df["shap"]],
        text=[f"{v:+.2f}" for v in plot_df["shap"]], textposition="outside",
    ))
    fig.update_layout(
        template=PIE_TEMPLATE, height=320, margin=dict(l=10, r=10, t=30, b=10),
        title="Вклад отдельных признаков в этот прогноз (≈ процентные пункты)",
        xaxis_title="Влияние на прогнозируемый % положительных результатов", yaxis_title=None,
    )
    st.plotly_chart(fig, width="stretch")

    st.caption(
        f"Baseline (средняя игра) ≈ {res['base_pct']:.1f}% · "
        f"прогноз для этой игры = {res['prediction_pct']:.1f}%. "
        "Вклады представляют собой SHAP-значения для линейной модели, пересчитанные в величины," \
        "приблизительно соответствующие процентным пунктам."
    )

    with st.expander("Веса признаков модели и глобальная значимость SHAP"):
        st.markdown("**Коэффициенты Lasso (модель линейная).** "
                    "Положительный вес → повышает рейтинг, отрицательный → понижает его.")
        w = ex.feature_weights(model, top=20).sort_values("weight")
        figw = go.Figure(go.Bar(
            x=w["weight"], y=w["feature"], orientation="h",
            marker_color=[GREEN if v > 0 else RED for v in w["weight"]],
        ))
        figw.update_layout(template=PIE_TEMPLATE, height=520,
                           margin=dict(l=10, r=10, t=10, b=10),
                           xaxis_title="Коэффициент") # (в лог-пространстве)
        st.plotly_chart(figw, width="stretch")

        st.markdown("**Глобальная значимость SHAP** (среднее значение SHAP по фоновой выборке).")
        gimp = ex.global_shap_importance(model, BG, top=15).sort_values("mean_abs_shap")
        figg = px.bar(gimp, x="mean_abs_shap", y="feature", orientation="h",
                      template=PIE_TEMPLATE)
        figg.update_traces(marker_color=ACCENT)
        figg.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10),
                           xaxis_title="среднее |SHAP|", yaxis_title=None)
        st.plotly_chart(figg, width="stretch")


# ---------------------------------------------------------------------------
#  Страница прогнозов

def build_record(form):
    """Сформируйте однострочный исходный DataFrame на основе данных из полей формы."""
    rec = {
        "name": form["name"],
        "release_date": str(form["release_date"]),
        "about_the_game": form["about_the_game"],
        "short_description": form["short_description"],
        "genres": form["genres"],
        "categories": form["categories"],
        "developers": [form["developer"]] if form["developer"] else [],
        "publishers": [form["publisher"]] if form["publisher"] else [],
        "full_audio_languages": form["audio_languages"],
        "mac": int(form["mac"]),
        "linux": int(form["linux"]),
        "os_count": 1 + int(form["mac"]) + int(form["linux"]),
        "lang_group": form["lang_group"],
        "lang_count": form["lang_count"],
        "has_ru": int(form["has_ru"]),
        "has_en": int(form["has_en"]),
    }
    for flag in META["content_flags"]:
        rec[flag] = int(form["flags"].get(flag, 0))
    return pd.DataFrame([rec])


def single_record_form():
    st.markdown("Заполните информацию об игре. Поля сгруппированы," \
    " чтобы внести изменения, откройте соответствующий раздел.")

    nd = META["numeric_defaults"]
    form = {"flags": {}}

    with st.expander("📋 Основные сведения", expanded=True):
        c1, c2 = st.columns(2)
        form["name"] = c1.text_input("Название игры", "My New Game")
        form["release_date"] = c2.date_input("Дата релиза", value=pd.Timestamp("2025-06-01"))
        form["about_the_game"] = st.text_area(
            "Об игре", "An epic adventure with deep story and soundtrack.")
        form["short_description"] = st.text_area(
            "Краткое описание", "Explore, fight and build your legend.")

    with st.expander("🎭 Жанры и категории", expanded=True):
        form["genres"] = st.multiselect(
            "Жанры", META["genre_choices"],
            default=[g for g in ["Action", "Adventure"] if g in META["genre_choices"]])
        form["categories"] = st.multiselect(
            "Steam категории", META["category_choices"],
            default=[c for c in ["Single-player", "Steam Achievements"]
                     if c in META["category_choices"]])
        st.markdown("**Особые индикаторы контента**")
        flag_labels = {
            "has_violence_gore": "Насилие / жестокость", "has_sexual_content": "Контент сексуального характера",
            "has_drugs_alcohol": "Наркотики* / алкоголь", "has_strong_language": "Ненормативная лексика",
            "has_mature_themes": "Темы для взрослых", "has_no_information": "Нет информации"
        }
        cols = st.columns(3)
        for i, flag in enumerate(META["content_flags"]):
            form["flags"][flag] = cols[i % 3].checkbox(
                flag_labels.get(flag, flag), value=bool(round(nd.get(flag, 0))))

    with st.expander("⚙️ Технические характеристики"):
        c1, c2, c3 = st.columns(3)
        form["mac"] = c1.checkbox("Поддержка MacOS", value=False)
        form["linux"] = c2.checkbox("Поддержка Linux", value=False)
        c3.markdown("Предполагается поддержка Windows.")
        c1, c2 = st.columns(2)
        form["lang_group"] = c1.selectbox("Языковая группа", META["lang_group_choices"])
        form["lang_count"] = c2.number_input(
            "Количество поддерживаемых языков", 1, 40, int(nd.get("lang_count", 1)))
        c1, c2 = st.columns(2)
        form["has_en"] = c1.checkbox("Поддержка английского языка", value=True)
        form["has_ru"] = c2.checkbox("Поддержка русского языка", value=False)

    with st.expander("📣 Маркетинговые данные"):
        c1, c2 = st.columns(2)
        dev_opts = ["(Неизвестный / Новый)"] + META["developer_choices"]
        pub_opts = ["(Неизвестный / Новый)"] + META["publisher_choices"]
        dev_sel = c1.selectbox("Разработчик", dev_opts)
        dev_custom = c1.text_input("…или введите имя разработчика", "")
        pub_sel = c2.selectbox("Издатель", pub_opts)
        pub_custom = c2.text_input("…или введите название издателя", "")
        form["developer"] = dev_custom or (dev_sel if dev_sel != "(Неизвестный / Новый)" else "")
        form["publisher"] = pub_custom or (pub_sel if pub_sel != "(Неизвестный / Новый)" else "")
        form["audio_languages"] = st.multiselect(
            "Набор языков аудиодорожки", META["audio_lang_choices"],
            default=[a for a in ["English"] if a in META["audio_lang_choices"]])

    if st.button("Прогнозирование рейтинга", type="primary", width="stretch"):
        X = build_record(form)
        pred = float(model.predict(X)[0])
        st.session_state["single_pred"] = (pred, X)

    if "single_pred" in st.session_state:
        pred, X = st.session_state["single_pred"]
        delta = pred - META["global_mean_rating"]
        c1, c2 = st.columns([1, 2])
        c1.metric("Прогнозируемые положительные отзывы", f"{pred:.1f}%",
                  f"{delta:+.1f} по сравнению со средней игрой")
        c2.progress(min(max(pred / 100, 0), 1.0))
        st.divider()
        st.subheader("Почему именно такой прогноз?")
        render_explanation(X)


def batch_form():
    st.markdown("Загрузите CSV-файл в том же формате, что и исходные данные "
                "(столбцы указаны в верхней части ноутбука). "
                "Для каждой строки будет рассчитан прогнозируемый рейтинг.")
    up = st.file_uploader("CSV файл", type=["csv"])
    if up is None:
        st.info("Ожидается загрузка CSV-файла.")
        return
    raw = pd.read_csv(up)
    st.write(f"Загружено **{len(raw)}** строк, **{raw.shape[1]}** колонок.")
    if st.button("Спрогнозировать все строки", type="primary"):
        preds = model.predict(raw)
        out = raw.copy()
        out["predicted_pct_positive"] = np.round(preds, 2)
        st.session_state["batch_out"] = out

    if "batch_out" in st.session_state:
        out = st.session_state["batch_out"]
        show_cols = [c for c in ["name", "predicted_pct_positive"] if c in out.columns]
        show_cols += [c for c in out.columns if c not in show_cols][:6]
        st.dataframe(out[["predicted_pct_positive"] +
                         [c for c in ["name"] if c in out.columns]].head(200),
                     width="stretch")

        fig = px.histogram(out, x="predicted_pct_positive", nbins=30,
                           template=PIE_TEMPLATE,
                           title="Распределение прогнозируемых рейтингов")
        fig.update_traces(marker_color=ACCENT)
        st.plotly_chart(fig, width="stretch")

        st.download_button(
            "⬇️ Скачать CSV-файл с прогнозами",
            out.to_csv(index=False).encode("utf-8"),
            "predictions.csv", "text/csv")

        st.divider()
        st.subheader("Объяснить одну строку")
        idx = st.number_input("Индекс строки для пояснения", 0, len(out) - 1, 0)
        render_explanation(raw.iloc[[int(idx)]])


def prediction_page():
    st.title("🎮 Инструмент для прогнозирования рейтинга игры в Steam до её выхода")
    st.caption(
        f"Модель: LassoCV · MAE на тесте ≈ {META['metrics']['mae']:.2f} pts · "
        f"прогнозирует долю положительных отзывов на основе предрелизных метаданных.")
    tab1, tab2 = st.tabs(["Одиночный прогноз", "Групповой прогноз (CSV)"])
    with tab1:
        single_record_form()
    with tab2:
        batch_form()


# --------------------------------------------------------------------------- #
#  EDA dashboard                           #
# --------------------------------------------------------------------------- #
def explode_on(df, list_col):
    sub = df[[list_col + "_list", "pct_pos_total", "year"]].explode(list_col + "_list")
    sub = sub.rename(columns={list_col + "_list": list_col})
    return sub.dropna(subset=[list_col])


def eda_page():
    st.title("📊 Исследовательская панель 🕵️")
    st.caption("Загрузите набор данных (в том же формате, что и для обучения) чтобы изучить, "
                "как влияет жанр, время и контент связан с рейтингом.")

    eda_up = st.file_uploader("Загрузить набор данных CSV ⬇️", type=["csv"], key="eda_csv")
    if eda_up is not None:
        df = read_eda_csv(eda_up.getvalue())
    else:
        df = load_eda()                              # bundled training data fallback
        if df is not None:
            st.info("Демонстрация набора данных, на котором обучалась модель. "
                    "Загрузите CSV файл выше, чтобы изучить другой файл.")

    if df is None:
        st.warning("Upload a CSV file to populate the dashboard.")
        st.stop()

    if pu.TARGET not in df.columns:
        st.error(f"Целевая переменная`{pu.TARGET}` не найдена, для нашборда необходима "
                    "колонка с рейтингом.")
        st.stop()

    target = pu.TARGET

    # ---- filters ---------------------------------------------------------- #
    ymin, ymax = int(df["year"].min()), int(df["year"].max())
    with st.sidebar:
        st.header("Фильтры панели мониторинга")
        yr = st.slider("Диапазон годов выпуска", ymin, ymax, (ymin, ymax))
        all_genres = sorted({g for lst in df["genres_list"] for g in lst})
        sel_genres = st.multiselect("Выделить жанры (динамика во времени)",
                                    all_genres, default=all_genres[:4])

    d = df[(df["year"] >= yr[0]) & (df["year"] <= yr[1])].copy()

    # ---- KPIs ------------------------------------------------------------- #
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Игры", f"{len(d):,}")
    k2.metric("Средний рейтинг", f"{d[target].mean():.1f}%")
    k3.metric("Медианный рейтинг", f"{d[target].median():.1f}%")
    k4.metric("Временной интервал (в годах)", f"{yr[0]}–{yr[1]}")
    st.divider()

    # ---- row 1: distribution + releases per year -------------------------- #
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Распределеине рейтинга")
        fig = px.histogram(d, x=target, nbins=40, template=PIE_TEMPLATE)
        fig.update_traces(marker_color=ACCENT)
        fig.update_layout(height=340, xaxis_title="% положительных отзывов",
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Релизы по годам")
        per_year = d.groupby("year").size().reset_index(name="games")
        fig = px.bar(per_year, x="year", y="games", template=PIE_TEMPLATE)
        fig.update_traces(marker_color=ACCENT)
        fig.update_layout(height=340, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    # ---- row 2: avg rating by genre + genre over time --------------------- #
    st.subheader("Как жанр влияет на рейтинг")
    ge = explode_on(d, "genres")
    genre_stats = (ge.groupby("genres")[target]
                   .agg(["mean", "count"]).reset_index()
                   .query("count >= 5").sort_values("mean", ascending=False))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Средний рейтинг по жанрам** (≥5 games)")
        fig = px.bar(genre_stats, x="mean", y="genres", orientation="h",
                     color="mean", color_continuous_scale="RdYlGn",
                     template=PIE_TEMPLATE, hover_data=["count"])
        fig.update_layout(height=460, yaxis={"categoryorder": "total ascending"},
                          coloraxis_showscale=False, xaxis_title="Средний % положительных отзывов",
                          yaxis_title=None, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**Динамика рейтинга жанра с течением времени**")
        gt = ge[ge["genres"].isin(sel_genres)] if sel_genres else ge
        trend = gt.groupby(["year", "genres"])[target].mean().reset_index()
        if trend.empty:
            st.caption("Выберите хотя бы один жанр на боковой панели.")
        else:
            fig = px.line(trend, x="year", y=target, color="genres",
                          markers=True, template=PIE_TEMPLATE)
            fig.update_layout(height=460, yaxis_title="Средний % положительных отзывов",
                              margin=dict(t=10, b=10))
            st.plotly_chart(fig, width="stretch")

    # heatmap genre x year
    st.markdown("**Тепловая карта «жанр - год»** (средний рейтинг)")
    top_g = genre_stats.head(12)["genres"].tolist()
    hm = (ge[ge["genres"].isin(top_g)]
          .groupby(["genres", "year"])[target].mean().reset_index())
    if not hm.empty:
        pivot = hm.pivot(index="genres", columns="year", values=target)
        fig = px.imshow(pivot, aspect="auto", color_continuous_scale="RdYlGn",
                        template=PIE_TEMPLATE, labels=dict(color="Средний %"))
        fig.update_layout(height=420, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    # ---- row 3: platform / language / content ----------------------------- #
    st.subheader("Другие факторы, влияющие на рейтинг")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Поддержка платформ**")
        rows = []
        for plat in ["mac", "linux"]:
            if plat in d.columns:
                for v, lbl in [(1, "Yes"), (0, "No")]:
                    sub = d[d[plat] == v]
                    if len(sub):
                        rows.append({"platform": plat, "support": lbl,
                                     "avg": sub[target].mean()})
        if rows:
            pf = pd.DataFrame(rows)
            fig = px.bar(pf, x="platform", y="avg", color="support",
                         barmode="group", template=PIE_TEMPLATE,
                         color_discrete_map={"Yes": GREEN, "No": "#999999"})
            fig.update_layout(height=340, yaxis_title="Средний % положительных отзывов",
                              margin=dict(t=10, b=10))
            st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**Рейтинг в зависимости от количества языков**")
        if "lang_count" in d.columns:
            d["lang_bin"] = pd.cut(d["lang_count"], [0, 1, 3, 6, 12, 100],
                                   labels=["1", "2-3", "4-6", "7-12", "13+"])
            lb = d.groupby("lang_bin")[target].mean().reset_index()
            fig = px.bar(lb, x="lang_bin", y=target, template=PIE_TEMPLATE)
            fig.update_traces(marker_color=ACCENT)
            fig.update_layout(height=340, yaxis_title="Средний % положительных отзывов",
                              xaxis_title="Языки", margin=dict(t=10, b=10))
            st.plotly_chart(fig, width="stretch")
    with c3:
        st.markdown("**Влияние описания контента**")
        flags = [c for c in d.columns if c.startswith("has_")
                 and c not in ("has_ru", "has_en", "has_no_information")]
        rows = []
        for f in flags:
            for v, lbl in [(1, "present"), (0, "absent")]:
                sub = d[d[f] == v]
                if len(sub):
                    rows.append({"flag": f.replace("has_", "").replace("_", " "),
                                 "state": lbl, "avg": sub[target].mean()})
        if rows:
            ff = pd.DataFrame(rows)
            fig = px.bar(ff, x="avg", y="flag", color="state", orientation="h",
                         barmode="group", template=PIE_TEMPLATE,
                         color_discrete_map={"present": RED, "absent": "#999999"})
            fig.update_layout(height=340, xaxis_title="Средний % положительных отзывов",
                              yaxis_title=None, margin=dict(t=10, b=10))
            st.plotly_chart(fig, width="stretch")

    # ---- row 4: top publishers ------------------------------------------- #
    st.subheader("Лучшие издатели по среднему рейтингу (≥5 игр)")
    pe = d[["publishers_list", target]].explode("publishers_list").dropna()
    pe = pe.rename(columns={"publishers_list": "publisher"})
    pub_stats = (pe.groupby("publisher")[target].agg(["mean", "count"])
                 .reset_index().query("count >= 5")
                 .sort_values("mean", ascending=False).head(15))
    if not pub_stats.empty:
        fig = px.bar(pub_stats, x="mean", y="publisher", orientation="h",
                     color="mean", color_continuous_scale="RdYlGn",
                     template=PIE_TEMPLATE, hover_data=["count"])
        fig.update_layout(height=460, yaxis={"categoryorder": "total ascending"},
                          coloraxis_showscale=False, xaxis_title="Средний % положительных отзывов",
                          yaxis_title=None, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
#  Router                                                                       #
# --------------------------------------------------------------------------- #
page = st.sidebar.radio("Навигация", ["Предсказание", "EDA Дашборд"])
st.sidebar.divider()
if page == "Предсказание":
    prediction_page()
else:
    eda_page()
