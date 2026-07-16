from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_predictor(texts, categories):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X, categories)

    return vectorizer, model


def predict_category(text, vectorizer, model):

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]

    probabilities = model.predict_proba(text_vector)[0]

    confidence = probabilities.max() * 100

    return prediction, confidence