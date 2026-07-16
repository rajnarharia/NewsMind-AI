import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("data/bbc-text.csv")

# Convert news text into TF-IDF features
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(df["text"])

# Create 5 clusters
model = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["cluster"] = model.fit_predict(X)

# Display results
print("News Clustering Completed Successfully!")
print("\nCluster Distribution:")
print(df["cluster"].value_counts().sort_index())

# Display top keywords of each cluster
terms = vectorizer.get_feature_names_out()
centers = model.cluster_centers_.argsort()[:, ::-1]

print("\nTop Keywords:")

for i in range(5):
    keywords = [terms[index] for index in centers[i, :10]]
    print(f"Cluster {i}: {', '.join(keywords)}")