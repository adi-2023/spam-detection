from flask import Flask, request, render_template
import joblib
import re
import os
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Initialize Flask app with explicit template folder
app = Flask(__name__, template_folder=os.path.abspath('templates'))

# Load model and vectorizer using absolute paths
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'spam_model.pkl')
tfidf_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf_vectorizer.pkl')

try:
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
except FileNotFoundError as e:
    raise RuntimeError(f"Model files not found. Please check paths: {e}")


def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'email' not in request.form:
        return "No email provided", 400

    email = request.form['email']
    cleaned_email = clean_text(email)
    vectorized_email = tfidf.transform([cleaned_email])
    prediction = model.predict(vectorized_email)[0]
    result = "SPAM" if prediction == 1 else "HAM"
    return render_template('index.html', prediction=result, email=email)


if __name__ == '__main__':
    app.run(debug=True, port=5000)