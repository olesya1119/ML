import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

from text_preprocessing import clean_text


def load_and_clean(path):
    df = pd.read_csv(path)
    df["text_clean"] = df["text"].fillna("").apply(clean_text)
    return df


def auto_keyword_detection(df, top_n=20):
    all_words = " ".join(df["text_clean"]).split()
    counter = Counter(all_words)
    top = counter.most_common(top_n)

    words = [w for w, c in top]
    counts = [c for w, c in top]

    plt.figure(figsize=(12, 4))
    plt.bar(words, counts)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Топ-{top_n} ключевых слов по частоте")
    plt.tight_layout()
    plt.savefig("rgr/data/auto_keywords.png", dpi=200)
    plt.close()

    print("\n=== Автоматически найденные ключевые слова ===")
    for w, c in top:
        print(f"{w}: {c}")

    return words


def tfidf_top(df, top_n=30):
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3)
    X = vec.fit_transform(df["text_clean"])
    terms = vec.get_feature_names_out()
    scores = X.mean(axis=0).A1

    idx = scores.argsort()[::-1][:top_n]
    df_top = pd.DataFrame({
        "term": terms[idx],
        "mean_tfidf": scores[idx]
    })

    df_top.to_csv("rgr/data/tfidf_top_terms.csv", index=False)

    print("\n=== Топ TF-IDF ===")
    print(df_top)

    return df_top


def run_nb(labeled_csv):
    df = pd.read_csv(labeled_csv)
    df["text_clean"] = df["text_clean"].fillna("")

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3)
    X = vec.fit_transform(df["text_clean"])
    y = df["label"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = MultinomialNB()
    clf.fit(Xtr, ytr)

    y_pred = clf.predict(Xte)

    print("\n=== Naive Bayes ===")
    print(classification_report(yte, y_pred))
    print("Матрица ошибок:\n", confusion_matrix(yte, y_pred))


if __name__ == "__main__":
    df = load_and_clean("rgr/data/vk_posts_raw.csv")

    # 1) АВТО-ВЫДЕЛЕНИЕ ключевых слов
    auto_keyword_detection(df, top_n=20)

    # 2) TF-IDF топ термины
    tfidf_top(df, top_n=30)

    # 3) Классификация (если есть файл разметки)
    try:
        run_nb("rgr/data/vk_posts_labeled.csv")
    except FileNotFoundError:
        print("\nФайл разметки не найден. Создай rgr/data/vk_posts_labeled.csv")
