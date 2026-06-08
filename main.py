import json
import re
import time
from datetime import datetime

import pandas as pd
import schedule
from google_play_scraper import reviews, Sort
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

APPS = {
    "payme": "uz.dida.payme",
    "click": "air.com.ssdsoftwaresolutions.clickuz",
    "uzum":  "uz.kapitalbank.android"
}

REVIEWS_PER_APP = 1000
RUN_EVERY_HOUR = 1
RESULTS_FILE    = "results.json"
TRAINING_CSV    = "labeled_reviews.csv"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zа-яё\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def train_model():
    print("[MODEL] Training on labeled_reviews.csv ...")
    df = pd.read_csv(TRAINING_CSV)
    df['review'] = df['review'].apply(clean_text)
    df = df.dropna(subset=['sentiment'])

    x = df['review']
    y = df['sentiment']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    uzbek_stops = [
        "va", "bu", "bir", "ham", "bor", "emas", "lekin", "yoki",
        "men", "sen", "u", "biz", "siz", "ular", "shu",
        "uchun", "bilan", "dan", "ga", "da", "ni", "ning",
        "juda", "hech", "endi", "keyin"
    ]

    vectorizer = TfidfVectorizer(stop_words=uzbek_stops, max_features=10000, ngram_range=(1, 2))
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec  = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(x_train_vec, y_train)

    predicted = model.predict(x_test_vec)
    print("[MODEL] Accuracy:", accuracy_score(y_test, predicted))
    print(classification_report(y_test, predicted, target_names=['negative', 'positive']))

    return model, vectorizer


# def scrape_reviews(app_name, app_id, count=100):
#     print(f"[SCRAPE] Fetching {count} reviews for {app_name} ...")
#     try:
#         result, _ = reviews(
#             app_id, lang='ru', country='uz', sort=Sort.NEWEST, count=count,
#         )
#         df = pd.DataFrame(result)[['content', 'score']]
#         df.columns = ['review', 'stars']
#         df['app'] = app_name
#         print(f"[SCRAPE] Got {len(df)} reviews for {app_name}")
#         return df
#     except Exception as e:
#         print(f"[SCRAPE] Failed for {app_name}: {e}")
#         return pd.DataFrame(columns=['review', 'stars', 'app'])
#
#
# def predict_batch(df, model, vectorizer):
#     df = df.copy()
#     df['review_clean'] = df['review'].apply(clean_text)
#     vectors = vectorizer.transform(df['review_clean'])
#     df['sentiment'] = model.predict(vectors)
#     return df
#
#
# def predict_sentiment(text, model, vectorizer):
#     text = clean_text(text)
#     vec  = vectorizer.transform([text])
#     pred = model.predict(vec)[0]
#     return "Positive 😊" if pred == 1 else "Negative 😡"
#
#
# def calculate_ranking(df):
#     results = []
#     for app in df['app'].unique():
#         app_df   = df[df['app'] == app]
#         total    = len(app_df)
#         positive = (app_df['sentiment'] == 1).sum()
#         negative = (app_df['sentiment'] == -1).sum()
#         score    = round(positive / total * 100, 1) if total > 0 else 0
#         results.append({
#             'app': app, 'total_reviews': int(total),
#             'positive': int(positive), 'negative': int(negative),
#             'positivity_score': score,
#         })
#
#     results_df = pd.DataFrame(results).sort_values(
#         'positivity_score', ascending=False
#     ).reset_index(drop=True)
#     results_df['rank'] = results_df.index + 1
#     return results_df
#
#
# def print_ranking(results_df, timestamp):
#     medals = ['🥇', '🥈', '🥉']
#     print("\n" + "="*55)
#     print(f"  FINTECH APP RANKINGS — {timestamp}")
#     print("="*55)
#     for _, row in results_df.iterrows():
#         medal   = medals[int(row['rank']) - 1]
#         bar_len = int(row['positivity_score'] / 2)
#         bar     = '█' * bar_len + '░' * (50 - bar_len)
#         print(f"\n{medal} #{int(row['rank'])} {row['app'].upper()}")
#         print(f"   Positivity : {row['positivity_score']}%")
#         print(f"   Positive   : {row['positive']} reviews")
#         print(f"   Negative   : {row['negative']} reviews")
#         print(f"   Total      : {row['total_reviews']} reviews")
#         print(f"   [{bar}]")
#     print("\n" + "="*55)
#     winner = results_df.iloc[0]
#     print(f"🏆 BEST APP: {winner['app'].upper()} ({winner['positivity_score']}% positive)")
#     print("="*55 + "\n")
#
#
# def save_results(results_df, timestamp):
#     data = {'last_updated': timestamp, 'rankings': results_df.to_dict(orient='records')}
#     with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)
#     print(f"[SAVE] Results saved to {RESULTS_FILE}")
#
#
# def run_pipeline(model, vectorizer):
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     print(f"\n[START] Pipeline run at {timestamp}")
#
#     all_reviews = []
#     for app_name, app_id in APPS.items():
#         df = scrape_reviews(app_name, app_id, count=REVIEWS_PER_APP)
#         if not df.empty:
#             all_reviews.append(df)
#
#     if not all_reviews:
#         print("[ERROR] No reviews scraped. Skipping this run.")
#         return
#
#     combined = pd.concat(all_reviews, ignore_index=True)
#     print(f"[MODEL] Predicting sentiment for {len(combined)} reviews ...")
#     combined = predict_batch(combined, model, vectorizer)
#
#     results_df = calculate_ranking(combined)
#     print_ranking(results_df, timestamp)
#     save_results(results_df, timestamp)
#     combined.to_csv('fresh_reviews_predicted.csv', index=False)
#     print(f"[SAVE] Raw predictions saved to fresh_reviews_predicted.csv")


# if __name__ == "__main__":
#     model, vectorizer = train_model()
#
#     # print(predict_sentiment("zo'r ilova, hammasi ajoyib ishlaydi", model, vectorizer))
#     # print(predict_sentiment("dahshatli ilova, umuman ishlamaydi, pulim yo'qoldi", model, vectorizer))
#     # print(predict_sentiment("yaxshi ilova lekin to'lov ham ishlayapti", model, vectorizer))
#     # print(predict_sentiment("отличное приложение, всё работает быстро", model, vectorizer))
#
#     run_pipeline(model, vectorizer)
#
#     schedule.every(RUN_EVERY_HOUR).hours.do(
#         run_pipeline, model=model, vectorizer=vectorizer
#     )
#
#     print(f"\n[SCHEDULER] Running every {RUN_EVERY_HOUR} hour(s). Press Ctrl+C to stop.\n")
#
#     while True:
#         schedule.run_pending()
#         time.sleep(60)