from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def find_similar_articles(articles, selected_index, top_n=5):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    tfidf_matrix = vectorizer.fit_transform(articles)

    similarity_scores = cosine_similarity(
        tfidf_matrix[selected_index],
        tfidf_matrix
    ).flatten()

    similar_indices = similarity_scores.argsort()[::-1]

    # Remove the selected article itself
    similar_indices = [
        index
        for index in similar_indices
        if index != selected_index
    ]

    return similar_indices[:top_n]