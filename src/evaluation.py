from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def evaluate_classifier(texts, categories):

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        categories,
        test_size=0.2,
        random_state=42,
        stratify=categories
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train_vectorized, y_train)

    predictions = model.predict(X_test_vectorized)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return accuracy