from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import pandas as pd


def create_cluster_visualization(texts, clusters):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1000
    )

    X = vectorizer.fit_transform(texts)

    # Reduce data to 2 dimensions
    pca = PCA(
        n_components=2,
        random_state=42
    )

    coordinates = pca.fit_transform(
        X.toarray()
    )

    visualization_df = pd.DataFrame({
        "PCA 1": coordinates[:, 0],
        "PCA 2": coordinates[:, 1],
        "Cluster": clusters
    })

    return visualization_df