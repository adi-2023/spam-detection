# 📧 Email Spam Detection System

A machine learning solution that classifies emails as **Spam** or **Ham** using Naive Bayes classifier, served via Flask web interface.

![image alt](https://github.com/adi-2023/spam-detection/blob/ac337fd4f372ace09991a1b7823a3af1754f65ee/1.png)
## Features
- NLP preprocessing with NLTK
- TF-IDF vectorization
- Naive Bayes classification (92%+ accuracy)
- Simple Flask web interface
- Lightweight & deployable

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
# Create virtual environment
python -m venv venv

# Activate (Unix/macOS)
source venv/bin/activate

# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Locally
```bash
# Preprocess data
python src/preprocess.py

# Train model
python src/train.py

# Launch web app
python src/app.py
```
Visit [http://localhost:5000](http://localhost:5000) in your browser.

![image alt](https://github.com/adi-2023/spam-detection/blob/ae01ed1926bd8d727bc7ab42099e9b51b8eb401c/2.png)

![image alt](https://github.com/adi-2023/spam-detection/blob/ae01ed1926bd8d727bc7ab42099e9b51b8eb401c/3.png)

## 🧠 How It Works
1. Messages are cleaned (lowercase, remove special chars)
2. NLTK tokenizes and stems words
3. TF-IDF converts text to numerical features
4. Pre-trained Naive Bayes model makes prediction

## 🛠️ Project Structure
```
email-spam-detector/
├── data/
│   ├── raw/spam.csv       # Original dataset
│   └── processed/         # Cleaned data
├── models/
│   └── spam_classifier.pkl  # Saved model
├── src/
│   ├── app.py            # Flask application
│   ├── preprocess.py     # Data cleaning
│   └── train.py          # Model training
└── templates/
    └── index.html        # Web interface
```

## 📚 Resources
- [Scikit-learn Docs](https://scikit-learn.org/stable/)
- [Flask Deployment Guide](https://flask.palletsprojects.com/en/2.2.x/deploying/)
- [NLTK Book](https://www.nltk.org/book/)

## 🤝 Contributing
PRs welcome! Please open an issue first to discuss changes.

---
```
