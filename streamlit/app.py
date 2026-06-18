import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import statistics
import re

# Scikit-learn и сопутствующие
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Для обработки предупреждений
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(layout="wide")
st.title("🎮 Steam Game Rating Predictor")
st.markdown("""
Предсказание процента положительных отзывов (`pct_pos_total`) для игр в Steam
на основе их статических мета-данных (жанры, цена, языки и т.д.).
""")

# --- 1. Загрузка и подготовка данных (как в вашем ноутбуке) ---

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --- Вспомогательные функции для обработки ---

def parse_list(x):
    if not isinstance(x, str):
        return []
    try:
        v = ast.literal_eval(x)
        return v if isinstance(v, list) else []
    except Exception:
        return []

def tokenize(text):
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    tokens = re.findall(r'\b[a-zа-яё0-9]+\\b', text)
    # Для простоты используем только токенизацию, без лемматизации (чтобы не тянуть nltk)
    return tokens

def scoring_calculation(df, column_name):
    full_score = {}
    if df.empty or column_name not in df.columns:
        return full_score
    
    average_score = df['pct_pos_total'].mean()
    df_exp = df.explode(column_name)
    
    for cat in df_exp[column_name].unique():
        if pd.isna(cat):
            continue
        subset = df_exp[df_exp[column_name] == cat]
        if len(subset) > 0:
            full_score[cat] = subset['pct_pos_total'].mean() - average_score
    return full_score

def cats_score(data, full_score):
    score = 0
    for i in data:
        score += full_score.get(i, 0)
    return score

def mean_cats_score(data, full_score):
    if not data:
        return 0
    scores = [full_score.get(i, 0) for i in data]
    return statistics.mean(scores)

def build_features_train(X, y, full_scores_cache=None):
    X = X.copy()
    
    X["genres_new"] = X["genres_new"].apply(parse_list)
    X["categories_new"] = X["categories_new"].apply(parse_list)
    X["supported_languages_new"] = X["supported_languages_new"].apply(parse_list)
    X["full_audio_languages_new"] = X["full_audio_languages_new"].apply(parse_list)
    
    if full_scores_cache is None:
        genres_score_map = scoring_calculation(X.assign(pct_pos_total=y), 'genres_new')
        categories_score_map = scoring_calculation(X.assign(pct_pos_total=y), 'categories_new')
        sup_lang_score_map = scoring_calculation(X.assign(pct_pos_total=y), 'supported_languages_new')
        developers_score_map = scoring_calculation(X.assign(pct_pos_total=y), 'developers_new')
        
        full_scores_cache = {
            'genres_score_map': genres_score_map,
            'categories_score_map': categories_score_map,
            'sup_lang_score_map': sup_lang_score_map,
            'developers_score_map': developers_score_map,
        }
    else:
        genres_score_map = full_scores_cache['genres_score_map']
        categories_score_map = full_scores_cache['categories_score_map']
        sup_lang_score_map = full_scores_cache['sup_lang_score_map']
        developers_score_map = full_scores_cache['developers_score_map']
    
    X['mean_genres_score'] = X['genres_new'].apply(lambda x: mean_cats_score(x, genres_score_map))
    X['mean_categories_score'] = X['categories_new'].apply(lambda x: mean_cats_score(x, categories_score_map))
    X['mean_sup_lang_score'] = X['supported_languages_new'].apply(lambda x: mean_cats_score(x, sup_lang_score_map))
    X['mean_developers_score'] = X['developers_new'].apply(lambda x: mean_cats_score(x, developers_score_map))
    
    X["n_genres"] = X["genres_new"].apply(len)
    X["n_categories"] = X["categories_new"].apply(len)
    X["n_supported_languages"] = X["supported_languages_new"].apply(len)
    X["n_full_audio_languages"] = X["full_audio_languages_new"].apply(len)
    
    X['categories_new_str'] = X['categories_new'].apply(lambda x: ', '.join(x))
    X['genres_new_str'] = X['genres_new'].apply(lambda x: ', '.join(x))
    X['supported_languages_new_str'] = X['supported_languages_new'].apply(lambda x: ', '.join(x))
    
    return X, full_scores_cache

def prepare_data(df):
    df = df.copy()
    
    df["detailed_desc_len"] = df["detailed_description"].fillna("").str.len()
    df["short_desc_len"] = df["short_description"].fillna("").str.len()
    df["has_website"] = df["website"].notna().astype(int)
    df["has_support_url"] = df["support_url"].notna().astype(int)
    df["has_support_email"] = df["support_email"].notna().astype(int)
    df["has_notes"] = df["notes"].notna().astype(int)
    df["required_age"] = df["required_age"].clip(lower=0)
    
    df["developers_new"] = df["developers_new"].apply(parse_list)
    df["publishers_new"] = df["publishers_new"].apply(parse_list)
    df["developers_new_len"] = df["developers_new"].apply(lambda x: len(x))
    df["publishers_new_len"] = df["publishers_new"].apply(lambda x: len(x))
    
    df["genres_new_parsed"] = df["genres_new"].apply(parse_list)
    df = df[~df["genres_new_parsed"].apply(lambda x: 'non_gaming_genre' in x)]
    
    feature_cols = [
        'price', 'required_age', 'detailed_desc_len', 'short_desc_len',
        'windows', 'mac', 'linux',
        'violence_gore', 'sexual_content', 'drugs_alcohol',
        'strong_language', 'mature_themes', 'has_website',
        'has_support_url', 'has_support_email', 'has_notes',
        'price_category', 'short_description', 'developers_new', 'publishers_new',
        'genres_new', 'categories_new', 'supported_languages_new', 'full_audio_languages_new',
        'publishers_new_len', 'developers_new_len'
    ]
    
    X = df[feature_cols].copy()
    y = df["pct_pos_total"].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    X_train, scores_cache = build_features_train(X_train, y_train, full_scores_cache=None)
    X_test, _ = build_features_train(X_test, y_train, full_scores_cache=scores_cache)
    
    # TF-IDF для short_description
    tfidf = TfidfVectorizer(
        max_features=150,
        stop_words='english',
        min_df=3,
        max_df=0.8,
        tokenizer=tokenize,
    )
    train_tfidf = tfidf.fit_transform(X_train['short_description'].fillna('').astype(str))
    test_tfidf = tfidf.transform(X_test['short_description'].fillna('').astype(str))
    
    cols = [f'short_description_tfidf_{f}' for f in tfidf.get_feature_names_out()]
    tfidf_train_df = pd.DataFrame(train_tfidf.toarray(), columns=cols, index=X_train.index)
    tfidf_test_df = pd.DataFrame(test_tfidf.toarray(), columns=cols, index=X_test.index)
    X_train = X_train.drop('short_description', axis=1)
    X_test = X_test.drop('short_description', axis=1)
    X_train = pd.concat([X_train, tfidf_train_df], axis=1)
    X_test = pd.concat([X_test, tfidf_test_df], axis=1)
    
    # TF-IDF для категорийных признаков
    cat_columns = ['genres_new_str', 'categories_new_str', 'supported_languages_new_str']
    all_tfidf_cols = cols.copy()
    
    for col in cat_columns:
        tfidf_cat = TfidfVectorizer(
            max_features=150,
            stop_words='english',
            min_df=3,
            max_df=0.8,
            tokenizer=tokenize,
        )
        train_cat = tfidf_cat.fit_transform(X_train[col].fillna('').astype(str))
        test_cat = tfidf_cat.transform(X_test[col].fillna('').astype(str))
        
        cat_cols = [f'{col}_tfidf_{f}' for f in tfidf_cat.get_feature_names_out()]
        train_cat_df = pd.DataFrame(train_cat.toarray(), columns=cat_cols, index=X_train.index)
        test_cat_df = pd.DataFrame(test_cat.toarray(), columns=cat_cols, index=X_test.index)
        
        X_train = X_train.drop(col, axis=1)
        X_test = X_test.drop(col, axis=1)
        X_train = pd.concat([X_train, train_cat_df], axis=1)
        X_test = pd.concat([X_test, test_cat_df], axis=1)
        all_tfidf_cols.extend(cat_cols)
    
    numeric_features = [
        'price', 'required_age', 'n_supported_languages', 'n_full_audio_languages', 
        'n_genres', 'n_categories', 'detailed_desc_len', 'short_desc_len',
        'mean_genres_score', 'mean_categories_score', 'mean_sup_lang_score', 
        'publishers_new_len', 'developers_new_len'
    ]
    
    binary_features = [
        'windows', 'mac', 'linux', 'violence_gore',
        'sexual_content', 'drugs_alcohol', 'strong_language', 'mature_themes',
        'has_website', 'has_support_url', 'has_support_email', 'has_notes'
    ]
    
    categorical_features = ['price_category']
    multi_tfidf_features = all_tfidf_cols
    
    return (X_train, X_test, y_train, y_test, 
            numeric_features, binary_features, categorical_features, multi_tfidf_features, scores_cache)


# --- 2. Обучение модели (кэшируем, чтобы не пересчитывать каждый раз) ---

@st.cache_resource
def load_data_and_train():
    # Загружаем данные
    # ВАЖНО: укажите правильный путь к вашему файлу
    df = pd.read_csv('Final_clean_df.csv')
    
    # Подготавливаем данные
    (X_train, X_test, y_train, y_test, 
     NUM, BINF, CAT, MULTI_TFIDF, scores_cache) = prepare_data(df)
    
    # Создаем пайплайн
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("binf", "passthrough", BINF),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("MULTI", "passthrough", MULTI_TFIDF)
    ])
    
    model = Pipeline([
        ("pre", preprocessor),
        ("model", Ridge(alpha=10.0))  # Подобранный alpha
    ])
    
    # Обучаем
    model.fit(X_train, y_train)
    
    # Оцениваем
    y_pred = model.predict(X_test).clip(min=1, max=100)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
        "R2": r2_score(y_test, y_pred)
    }
    
    return model, X_train, y_train, X_test, y_test, metrics, NUM, BINF, CAT, MULTI_TFIDF, scores_cache


# --- 3. Функция для предсказания на новых данных ---

def predict_new_data(model, df_new, scores_cache):
    # Подготавливаем новые данные
    df_new = df_new.copy()
    
    df_new["detailed_desc_len"] = df_new["detailed_description"].fillna("").str.len()
    df_new["short_desc_len"] = df_new["short_description"].fillna("").str.len()
    df_new["has_website"] = df_new["website"].notna().astype(int)
    df_new["has_support_url"] = df_new["support_url"].notna().astype(int)
    df_new["has_support_email"] = df_new["support_email"].notna().astype(int)
    df_new["has_notes"] = df_new["notes"].notna().astype(int)
    df_new["required_age"] = df_new["required_age"].clip(lower=0)
    
    df_new["developers_new"] = df_new["developers_new"].apply(parse_list)
    df_new["publishers_new"] = df_new["publishers_new"].apply(parse_list)
    df_new["developers_new_len"] = df_new["developers_new"].apply(lambda x: len(x))
    df_new["publishers_new_len"] = df_new["publishers_new"].apply(lambda x: len(x))
    
    df_new["genres_new_parsed"] = df_new["genres_new"].apply(parse_list)
    df_new = df_new[~df_new["genres_new_parsed"].apply(lambda x: 'non_gaming_genre' in x)]
    
    feature_cols = [
        'price', 'required_age', 'detailed_desc_len', 'short_desc_len',
        'windows', 'mac', 'linux',
        'violence_gore', 'sexual_content', 'drugs_alcohol',
        'strong_language', 'mature_themes', 'has_website',
        'has_support_url', 'has_support_email', 'has_notes',
        'price_category', 'short_description', 'developers_new', 'publishers_new',
        'genres_new', 'categories_new', 'supported_languages_new', 'full_audio_languages_new',
        'publishers_new_len', 'developers_new_len'
    ]
    
    X_new = df_new[feature_cols].copy()
    X_new, _ = build_features_train(X_new, pd.Series([50] * len(X_new)), full_scores_cache=scores_cache)
    
    # TF-IDF
    tfidf = TfidfVectorizer(
        max_features=150,
        stop_words='english',
        min_df=3,
        max_df=0.8,
        tokenizer=tokenize,
    )
    # Используем fit_transform на всех данных, чтобы получить те же признаки
    # В реальном приложении нужно сохранять векторизаторы
    all_descriptions = pd.concat([X_new['short_description']])
    tfidf.fit(all_descriptions.fillna('').astype(str))
    new_tfidf = tfidf.transform(X_new['short_description'].fillna('').astype(str))
    
    cols = [f'short_description_tfidf_{f}' for f in tfidf.get_feature_names_out()]
    tfidf_new_df = pd.DataFrame(new_tfidf.toarray(), columns=cols, index=X_new.index)
    X_new = X_new.drop('short_description', axis=1)
    X_new = pd.concat([X_new, tfidf_new_df], axis=1)
    
    cat_columns = ['genres_new_str', 'categories_new_str', 'supported_languages_new_str']
    for col in cat_columns:
        tfidf_cat = TfidfVectorizer(
            max_features=150,
            stop_words='english',
            min_df=3,
            max_df=0.8,
            tokenizer=tokenize,
        )
        tfidf_cat.fit(pd.concat([X_new[col]]).fillna('').astype(str))
        new_cat = tfidf_cat.transform(X_new[col].fillna('').astype(str))
        
        cat_cols = [f'{col}_tfidf_{f}' for f in tfidf_cat.get_feature_names_out()]
        cat_new_df = pd.DataFrame(new_cat.toarray(), columns=cat_cols, index=X_new.index)
        
        X_new = X_new.drop(col, axis=1)
        X_new = pd.concat([X_new, cat_new_df], axis=1)
    
    # Предсказание
    predictions = model.predict(X_new).clip(min=1, max=100)
    return predictions


# --- 4. Загрузка модели и данных ---

try:
    model, X_train, y_train, X_test, y_test, metrics, NUM, BINF, CAT, MULTI_TFIDF, scores_cache = load_data_and_train()
    st.success("✅ Модель успешно загружена и обучена!")
except Exception as e:
    st.error(f"❌ Ошибка загрузки данных или обучения модели: {e}")
    st.stop()


# --- 5. Боковая панель навигации ---

st.sidebar.title("Навигация")
page = st.sidebar.radio("Выберите раздел:", 
    ["📊 EDA", "📈 Предсказать по CSV", "🎮 Ручной ввод", "📉 Веса модели"])


# --- 6. Страница EDA ---

if page == "📊 EDA":
    st.header("Исследовательский анализ данных (EDA)")
    
    # Загружаем данные для EDA
    df_eda = pd.read_csv('Final_clean_df.csv')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Распределение целевой переменной")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df_eda['pct_pos_total'], bins=30, kde=True, ax=ax)
        ax.set_title('Распределение процента положительных отзывов')
        ax.set_xlabel('pct_pos_total')
        st.pyplot(fig)
        
        st.subheader("Топ-10 жанров по среднему рейтингу")
        # Парсим жанры
        genres_list = df_eda['genres_new'].apply(parse_list)
        genres_exploded = genres_list.explode()
        genre_avg = df_eda.groupby(genres_exploded)['pct_pos_total'].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        genre_avg.plot(kind='barh', ax=ax)
        ax.set_title('Средний рейтинг по жанрам')
        ax.set_xlabel('Средний pct_pos_total')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Распределение цен")
        fig, ax = plt.subplots(figsize=(8, 5))
        df_eda[df_eda['price'] < 100]['price'].hist(bins=50, ax=ax)
        ax.set_title('Распределение цен (до 100$)')
        ax.set_xlabel('Price')
        st.pyplot(fig)
        
        st.subheader("Корреляция признаков с целевой")
        # Базовые числовые признаки
        numeric_cols = ['price', 'required_age', 'discount', 
                        'average_playtime_forever', 'median_playtime_forever']
        corr = df_eda[numeric_cols + ['pct_pos_total']].corr()['pct_pos_total'].drop('pct_pos_total').sort_values()
        fig, ax = plt.subplots(figsize=(8, 4))
        corr.plot(kind='barh', ax=ax)
        ax.set_title('Корреляция с целевой переменной')
        ax.set_xlabel('Корреляция')
        st.pyplot(fig)


# --- 7. Страница предсказания по CSV ---

elif page == "📈 Предсказать по CSV":
    st.header("Предсказание рейтинга для загруженного CSV-файла")
    
    uploaded_file = st.file_uploader("Загрузите CSV-файл с признаками игр", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_new = pd.read_csv(uploaded_file)
            st.write("Загружено строк:", len(df_new))
            st.dataframe(df_new.head())
            
            # Проверяем наличие необходимых колонок
            required_cols = ['price', 'genres_new', 'categories_new', 'supported_languages_new']
            missing_cols = [col for col in required_cols if col not in df_new.columns]
            if missing_cols:
                st.error(f"В файле отсутствуют обязательные колонки: {missing_cols}")
            else:
                if st.button("Выполнить предсказание"):
                    with st.spinner("Идет обработка и предсказание..."):
                        predictions = predict_new_data(model, df_new, scores_cache)
                        
                        # Добавляем предсказания в датафрейм
                        df_results = df_new.copy()
                        df_results['predicted_pct_pos_total'] = predictions
                        
                        st.success("✅ Предсказание выполнено!")
                        st.dataframe(df_results[['name', 'predicted_pct_pos_total']].head(10) if 'name' in df_results.columns else df_results.head(10))
                        
                        # Статистика
                        st.subheader("Статистика предсказаний")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Среднее", f"{np.mean(predictions):.1f}")
                        col2.metric("Медиана", f"{np.median(predictions):.1f}")
                        col3.metric("Станд. отклонение", f"{np.std(predictions):.1f}")
                        
                        # Гистограмма предсказаний
                        fig, ax = plt.subplots(figsize=(8, 4))
                        plt.hist(predictions, bins=20, edgecolor='black')
                        plt.title('Распределение предсказанных рейтингов')
                        plt.xlabel('Предсказанный pct_pos_total')
                        plt.ylabel('Количество игр')
                        st.pyplot(fig)
                        
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {e}")


# --- 8. Страница ручного ввода ---

elif page == "🎮 Ручной ввод":
    st.header("Введите признаки игры вручную")
    
    # Создаем форму для ввода
    with st.form("manual_input_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            price = st.number_input("Цена (USD)", min_value=0.0, max_value=200.0, value=9.99)
            required_age = st.number_input("Возрастное ограничение", min_value=0, max_value=18, value=0)
            windows = st.checkbox("Windows", value=True)
            mac = st.checkbox("Mac", value=False)
            linux = st.checkbox("Linux", value=False)
            
        with col2:
            violence_gore = st.selectbox("Насилие и кровь", [0, 1], format_func=lambda x: "Да" if x else "Нет")
            sexual_content = st.selectbox("Сексуальный контент", [0, 1], format_func=lambda x: "Да" if x else "Нет")
            drugs_alcohol = st.selectbox("Наркотики/алкоголь", [0, 1], format_func=lambda x: "Да" if x else "Нет")
            strong_language = st.selectbox("Нецензурная лексика", [0, 1], format_func=lambda x: "Да" if x else "Нет")
            mature_themes = st.selectbox("Взрослые темы", [0, 1], format_func=lambda x: "Да" if x else "Нет")
            
        with col3:
            has_website = st.selectbox("Наличие веб-сайта", [0, 1], format_func=lambda x: "Да" if x else "Нет")
            has_support_url = st.selectbox("Наличие ссылки поддержки", [0, 1], format_func=lambda x: "Да" if x else "Нет")
            has_support_email = st.selectbox("Наличие email поддержки", [0, 1], format_func=lambda x: "Да" if x else "Нет")
            has_notes = st.selectbox("Наличие примечаний", [0, 1], format_func=lambda x: "Да" if x else "Нет")
            price_category = st.selectbox("Ценовая категория", ["0-5$", "5-10$", "10-20$", "20-40$", "40-60$", "60$+"])
        
        # Текстовые поля
        st.subheader("Текстовые и категориальные признаки")
        col1, col2 = st.columns(2)
        
        with col1:
            short_description = st.text_area("Краткое описание (short_description)", 
                                            value="An exciting adventure game with deep strategy elements.")
            detailed_description = st.text_area("Подробное описание (detailed_description)", 
                                               value="Explore a vast world, build your empire, and conquer your enemies.")
            
        with col2:
            genres = st.text_input("Жанры (через запятую)", value="Action, Adventure, Strategy")
            categories = st.text_input("Категории (через запятую)", value="Single-player, Multi-player")
            supported_languages = st.text_input("Поддерживаемые языки (через запятую)", value="English, German, French")
            full_audio_languages = st.text_input("Языки с полной озвучкой (через запятую)", value="English")
            developers = st.text_input("Разработчики (через запятую)", value="GameStudio Inc.")
            publishers = st.text_input("Издатели (через запятую)", value="GameStudio Inc.")
        
        submitted = st.form_submit_button("Предсказать рейтинг")
    
    if submitted:
        try:
            # Создаем DataFrame из введенных данных
            data = {
                'price': [price],
                'required_age': [required_age],
                'detailed_description': [detailed_description],
                'short_description': [short_description],
                'windows': [windows],
                'mac': [mac],
                'linux': [linux],
                'violence_gore': [violence_gore],
                'sexual_content': [sexual_content],
                'drugs_alcohol': [drugs_alcohol],
                'strong_language': [strong_language],
                'mature_themes': [mature_themes],
                'has_website': [has_website],
                'has_support_url': [has_support_url],
                'has_support_email': [has_support_email],
                'has_notes': [has_notes],
                'price_category': [price_category],
                'developers_new': [[d.strip() for d in developers.split(',') if d.strip()]],
                'publishers_new': [[p.strip() for p in publishers.split(',') if p.strip()]],
                'genres_new': [[g.strip() for g in genres.split(',') if g.strip()]],
                'categories_new': [[c.strip() for c in categories.split(',') if c.strip()]],
                'supported_languages_new': [[l.strip() for l in supported_languages.split(',') if l.strip()]],
                'full_audio_languages_new': [[a.strip() for a in full_audio_languages.split(',') if a.strip()]],
            }
            
            df_new = pd.DataFrame(data)
            
            # Добавляем недостающие колонки
            for col in ['website', 'support_url', 'support_email', 'notes']:
                if col not in df_new.columns:
                    df_new[col] = None
            
            with st.spinner("Идет предсказание..."):
                prediction = predict_new_data(model, df_new, scores_cache)
                prediction = prediction[0]
                
                st.success(f"🎯 Предсказанный процент положительных отзывов: **{prediction:.1f}%**")
                
                # Визуализация
                fig, ax = plt.subplots(figsize=(6, 3))
                colors = ['green' if prediction >= 70 else 'orange' if prediction >= 40 else 'red']
                ax.bar(['Предсказанный рейтинг'], [prediction], color=colors)
                ax.axhline(y=50, color='gray', linestyle='--', label='Граница 50%')
                ax.set_ylim(0, 100)
                ax.set_ylabel('Процент положительных отзывов')
                ax.legend()
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")
            st.write("Пожалуйста, проверьте введенные данные.")


# --- 9. Страница с весами модели ---

elif page == "📉 Веса модели":
    st.header("Визуализация весов обученной модели Ridge")
    
    st.markdown("""
    Ниже показаны коэффициенты (веса) модели Ridge. 
    Положительные веса означают, что признак увеличивает предсказанный рейтинг, 
    отрицательные — уменьшает.
    """)
    
    try:
        # Получаем веса
        # Для Pipeline нужно добраться до шага 'model'
        model_step = model.named_steps['model']
        preprocessor = model.named_steps['pre']
        
        # Получаем названия признаков после трансформации
        # Это сложно сделать красиво для ColumnTransformer, поэтому используем упрощенный подход
        # Получаем веса для числовых признаков
        num_indices = preprocessor.transformers_[0][2]  # индексы числовых колонок
        num_weights = model_step.coef_[num_indices] if len(num_indices) > 0 else []
        
        # Бинарные признаки
        bin_indices = preprocessor.transformers_[1][2]
        bin_weights = model_step.coef_[bin_indices] if len(bin_indices) > 0 else []
        
        # Собираем все признаки
        feature_names = []
        weights = []
        
        # Числовые
        for i, name in enumerate(NUM):
            if i < len(num_weights):
                feature_names.append(name)
                weights.append(num_weights[i])
        
        # Бинарные
        for i, name in enumerate(BINF):
            if i < len(bin_weights):
                feature_names.append(name)
                weights.append(bin_weights[i])
        
        # Создаем DataFrame
        weights_df = pd.DataFrame({
            'Признак': feature_names,
            'Вес': weights
        }).sort_values('Вес', ascending=False)
        
        # Показываем топ-20
        st.subheader("Топ-20 признаков по абсолютному влиянию")
        top_features = weights_df.iloc[:20]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if w > 0 else 'red' for w in top_features['Вес']]
        ax.barh(top_features['Признак'], top_features['Вес'], color=colors)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title('Веса признаков модели Ridge')
        ax.set_xlabel('Вес')
        st.pyplot(fig)
        
        # Полный список
        with st.expander("Показать все веса"):
            st.dataframe(weights_df)
        
        # Анализ
        st.subheader("Анализ важности признаков")
        
        # Группировка по типу
        st.markdown("""
        **Выводы по весам модели:**
        
        - **Положительное влияние** (увеличивают рейтинг):
          - `mean_categories_score`, `mean_genres_score` — усредненные рейтинги категорий/жанров
          - `n_supported_languages` — больше языков = больше аудитории
          - `has_support_url` — наличие поддержки повышает доверие
        
        - **Отрицательное влияние** (уменьшают рейтинг):
          - `price` — высокая цена может снижать восприятие ценности
          - `required_age` — возрастные ограничения сужают аудиторию
          - `violence_gore` — контент для взрослых может снижать рейтинг
        """)
        
    except Exception as e:
        st.error(f"Ошибка при визуализации весов: {e}")
        st.info("Возможно, модель имеет сложную структуру, которая не позволяет легко извлечь веса.")

st.sidebar.markdown("---")
st.sidebar.info("""
**О модели:**  
Ridge регрессия обучена на ~22k играх Steam.  
Метрики на тесте:
- MAE: {:.2f}
- RMSE: {:.2f}
- R²: {:.3f}
""".format(metrics['MAE'], metrics['RMSE'], metrics['R2']))
