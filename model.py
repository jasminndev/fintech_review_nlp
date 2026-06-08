import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import re
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("reviews_labeled.csv")
print("Dataset size:", df.shape)
print(df.head())
print(df['app'].value_counts())


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zа-яё\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['review'] = df['review'].apply(clean_text)

train_df = pd.read_csv("reviews_labeled.csv")
train_df['review'] = train_df['review'].apply(clean_text)

X = train_df['review']
y = train_df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

uzbek_stops = [
    "va", "bu", "bir", "ham", "bor", "emas", "lekin", "yoki",
    "men", "sen", "u", "biz", "siz", "ular", "shu",
    "uchun", "bilan", "dan", "ga", "da", "ni", "ning",
    "juda", "hech", "endi", "keyin"
]

tfidf = TfidfVectorizer(
    stop_words=uzbek_stops,
    max_features=10000,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)


lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train_tfidf, y_train)
lr_pred = lr.predict(X_test_tfidf)

print("LR Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred, target_names=['negative', 'positive']))


nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
nb_pred = nb.predict(X_test_tfidf)

print("NB Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred, target_names=['negative', 'positive']))

svm = LinearSVC(max_iter=2000, class_weight='balanced')
svm.fit(X_train_tfidf, y_train)
svm_pred = svm.predict(X_test_tfidf)

print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred, target_names=['negative', 'positive']))


results = pd.DataFrame({
    "Model": ["Logistic Regression", "Naive Bayes", "SVM"],
    "Accuracy": [
        accuracy_score(y_test, lr_pred),
        accuracy_score(y_test, nb_pred),
        accuracy_score(y_test, svm_pred)
    ]
})
print(results.sort_values("Accuracy", ascending=False))


best_model = lr
print(f"\nUsing: Logistic Regression")

df['predicted_sentiment'] = best_model.predict(tfidf.transform(df['review']))


ranking = (
    df.groupby('app')['predicted_sentiment']
    .value_counts(normalize=True)
    .unstack()
    .fillna(0)
    .round(3)
)
print(ranking)


print(ranking.columns.tolist())
print(ranking.head())

ranking = ranking.sort_values(by='Positive', ascending=False)
ranking['Rank'] = range(1, len(ranking) + 1)

print("\n=== FINTECH APP RANKING ===")
for i, (company, row) in enumerate(ranking.iterrows()):
    medals = ['🥇', '🥈', '🥉']
    print(f"{medals[i]} #{i+1} {company.upper()} — {row['Positive']*100:.1f}% Positive")

print(f"\n🏆 BEST APP: {ranking.index[0].upper()}")

positive_reviews = df[df['predicted_sentiment'] == 'Positive']
negative_reviews = df[df['predicted_sentiment'] == 'Negative']


cv = CountVectorizer(stop_words=uzbek_stops, max_features=50)
X_pos = cv.fit_transform(positive_reviews['review'])

freq = pd.DataFrame({
    'word':  cv.get_feature_names_out(),
    'count': X_pos.sum(axis=0).A1
}).sort_values('count', ascending=False)

print("\n=== TOP 20 POSITIVE KEYWORDS ===")
print(freq.head(20).to_string(index=False))


print("\n=== KEYWORDS PER COMPANY ===")
for company in ['payme', 'click', 'uzum']:
    company_pos = df[
        (df['app'].str.lower() == company) &
        (df['predicted_sentiment'] == 'Positive')
        ]
    if len(company_pos) == 0:
        continue

    cv_company = CountVectorizer(stop_words=uzbek_stops, max_features=20)
    X_company  = cv_company.fit_transform(company_pos['review'])

    freq_company = pd.DataFrame({
        'word':  cv_company.get_feature_names_out(),
        'count': X_company.sum(axis=0).A1
    }).sort_values('count', ascending=False)

    print(f"\n📱 {company.upper()} — Top 10 Positive Keywords:")
    print(freq_company.head(10).to_string(index=False))




