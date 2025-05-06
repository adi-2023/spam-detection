import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK stopwords (run once)
import nltk
nltk.download('stopwords')

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabets
    text = text.lower().split()
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# Load and clean data
df = pd.read_csv('../data/spam.csv', encoding='latin-1')
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
df['text'] = df['text'].apply(clean_text)
df.to_csv('../data/cleaned_spam.csv', index=False)