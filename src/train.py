import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib

# Load and clean data
df = pd.read_csv('../data/cleaned_spam.csv')
df = df.dropna(subset=['text'])  # Critical fix!

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['text'])
y = df['label'].map({'ham': 0, 'spam': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, '../models/spam_model.pkl')
joblib.dump(tfidf, '../models/tfidf_vectorizer.pkl')