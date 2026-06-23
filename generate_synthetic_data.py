"""Сгенерируйте синтетический файл clean_data_v1.csv, соответствующий схеме из ноутбука.

ТОЛЬКО ДЛЯ ЛОКАЛЬНОГО ТЕСТИРОВАНИЯ. 
Для получения реальных результатов заменить на настоящий файл clean_data_v1.csv.
Целевая переменная (pct_pos_total) сделана слабо зависимой от нескольких признаков, чтобы
модели было чему учиться, а графики SHAP выглядели осмысленно.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 2500

GENRES = ["Action", "Adventure", "RPG", "Strategy", "Casual", "Indie", "Simulation",
          "Sports", "Racing", "Puzzle", "Horror", "Shooter", "Platformer", "Rare1", "Rare2"]
CATS = ["Single-player", "Multi-player", "Co-op", "Steam Cloud", "Steam Trading Cards",
        "Full controller support", "Steam Achievements", "Captions available",
        "Family Sharing", "Cross-Platform Multiplayer", "Online PvP", "RareCat"]
DEVS = [f"Studio_{i}" for i in range(120)] + ["Indie Dev"] * 40
PUBS = [f"Pub_{i}" for i in range(90)] + ["Self Published"] * 60
LANGS = ["English", "Russian", "German", "French", "Spanish", "Japanese", "Chinese"]
LANG_GROUPS = ["1 язык", "2-5", "6-10", "11-20", "20+"]

rows = []
for i in range(N):
    n_g = rng.integers(1, 4)
    genres = list(rng.choice(GENRES, size=n_g, replace=False))
    n_c = rng.integers(2, 7)
    cats = list(rng.choice(CATS, size=n_c, replace=False))
    n_audio = rng.integers(1, 6)
    audio = list(rng.choice(LANGS, size=n_audio, replace=False))
    dev = rng.choice(DEVS)
    pub = rng.choice(PUBS)
    lang_count = int(rng.integers(1, 25))
    os_count = int(rng.integers(1, 4))
    mac = int(os_count >= 2 and rng.random() > 0.4)
    linux = int(os_count >= 3 and rng.random() > 0.5)
    date = pd.Timestamp("2009-01-01") + pd.Timedelta(days=int(rng.integers(0, 5900)))

    flags = {f"has_{k}": int(rng.random() < p) for k, p in [
        ("violence_gore", 0.25), ("sexual_content", 0.1), ("drugs_alcohol", 0.08),
        ("strong_language", 0.2), ("mature_themes", 0.18), ("no_information", 0.3)]}
    has_ru = int("Russian" in audio or rng.random() < 0.3)
    has_en = int("English" in audio or rng.random() < 0.8)

    # latent quality -> target, with signal from a few features
    base = 75
    base += 6 if "Multi-player" in cats else 0
    base += 5 if "Steam Cloud" in cats else 0
    base += 4 if "Full controller support" in cats else 0
    base += 3 if "RPG" in genres else 0
    base -= 5 if "Casual" in genres else 0
    base += 0.3 * lang_count
    base += (hash(dev) % 11) - 5          # pseudo developer reputation
    base += (hash(pub) % 9) - 4
    base += rng.normal(0, 9)
    pct = float(np.clip(base, 5, 99))

    rows.append({
        "appid": 1000 + i,
        "name": f"Game {i}",
        "release_date": date.strftime("%Y-%m-%d"),
        "required_age": int(rng.choice([0, 0, 0, 13, 17])),
        "price": round(float(rng.choice([0, 4.99, 9.99, 19.99, 29.99])), 2),
        "dlc_count": int(rng.integers(0, 5)),
        "about_the_game": rng.choice([
            "An epic adventure with stunning visuals and deep soundtrack.",
            "Survive in a harsh open world full of weapons and danger.",
            "A relaxing casual puzzle experience for the whole family.",
            "Fast paced shooter with online multiplayer and ranked play.",
            "Strategy game where you build, manage and conquer territories.",
        ]),
        "short_description": rng.choice([
            "Build, fight, explore.", "Love, heroes and visual storytelling.",
            "Hardcore survival.", "Casual fun for everyone.", "Competitive online battles.",
        ]),
        "mac": mac, "linux": linux,
        "full_audio_languages": str(audio),
        "developers": str([dev]),
        "publishers": str([pub]),
        "categories": str(cats),
        "genres": str(genres),
        "lang_group": str(rng.choice(LANG_GROUPS)),
        "lang_count": lang_count,
        "os_count": os_count,
        "has_ru": has_ru, "has_en": has_en,
        "pct_pos_total": pct,
        "num_reviews_total": int(rng.integers(10, 50000)),
        "year": date.year,
        **flags,
    })

pd.DataFrame(rows).to_csv("clean_data_v1.csv", index=False)
print(f"Wrote clean_data_v1.csv with {N} rows")
