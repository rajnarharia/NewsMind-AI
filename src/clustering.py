from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def create_clusters(texts, n_clusters=5):

    # Convert news articles into TF-IDF features
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    X = vectorizer.fit_transform(texts)

    # Create K-Means model
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    # Generate clusters
    clusters = model.fit_predict(X)
    score = silhouette_score(X, clusters)

    # Get feature names
    terms = vectorizer.get_feature_names_out()

    # Sort cluster centers to find top keywords
    centers = model.cluster_centers_.argsort()[:, ::-1]

    # Store top keywords
    keywords = {}

    for i in range(n_clusters):
        keywords[i] = [
            terms[index]
            for index in centers[i, :10]
        ]

    # Automatically assign topic names
    topic_names = {}

    for cluster_id, words in keywords.items():

        word_text = " ".join(words).lower()

        if any(word in word_text for word in [
            "game", "team", "player", "cup", "match",
            "football", "sport"
        ]):
            topic_names[cluster_id] = "Sports"

        elif any(word in word_text for word in [
            "government", "minister", "election",
            "party", "political", "labour"
        ]):
            topic_names[cluster_id] = "Politics"

        elif any(word in word_text for word in [
            "technology", "software", "computer",
            "mobile", "digital", "internet"
        ]):
            topic_names[cluster_id] = "Technology"

        elif any(word in word_text for word in [
            "film", "music", "show", "actor",
            "tv", "movie"
        ]):
            topic_names[cluster_id] = "Entertainment"

        else:
            topic_names[cluster_id] = "Business"

    return clusters, keywords, topic_names, score