# FAKE NEWS MODEL TRAINING + SAVING
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.utils import resample

# Load datasets
true_df = pd.read_csv('True.csv')
false_df = pd.read_csv('Fake.csv')

# Label encoding
true_df['label'] = 0
false_df['label'] = 1


# Merge title + content
true_df['content'] = true_df['title'].astype(str) + " " + true_df['text'].astype(str)
false_df['content'] = false_df['title'].astype(str) + " " + false_df['text'].astype(str)

# Balance
min_len = min(len(true_df), len(false_df))
true_df = resample(true_df, replace=False, n_samples=min_len, random_state=42)
false_df = resample(false_df, replace=False, n_samples=min_len, random_state=42)

# Combine and shuffle
df = pd.concat([true_df, false_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df['content'], df['label'], test_size=0.2, random_state=42)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.85,
    min_df=3,
    ngram_range=(1, 3),
    max_features=10000
)
tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(tfidf_train, y_train)

# Find best threshold
probs = model.predict_proba(tfidf_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_threshold = thresholds[np.argmax(f1_scores)]
adjusted_threshold = max(best_threshold, 0.72)

# Save model, vectorizer, threshold
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(adjusted_threshold, 'optimal_threshold.pkl')
