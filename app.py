from src.evaluation import evaluate_classifier
from src.predictor import train_predictor, predict_category
from src.visualization import create_cluster_visualization
import streamlit as st
import pandas as pd

from src.clustering import create_clusters
from src.similarity import find_similar_articles
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="NewsMind AI",
    page_icon="🧠",
    layout="wide"
)
with open("assets/style.css", encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/bbc-text.csv")


df = load_data()
classifier_accuracy = evaluate_classifier(
    df["text"],
    df["category"]
)
predictor_vectorizer, predictor_model = train_predictor(
    df["text"],
    df["category"]
)


# --------------------------------------------------
# CREATE AI CLUSTERS
# --------------------------------------------------

clusters, cluster_keywords, topic_names, silhouette_score_value = (
    create_clusters(df["text"])
)

df["cluster"] = clusters


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🧠 NewsMind AI")

st.subheader(
    "AI-Powered News Clustering & Topic Discovery Platform"
)

st.write(
    "Explore news articles, discover hidden topics, analyze AI-generated "
    "clusters, search articles, and find similar news using Machine Learning."
)


# --------------------------------------------------
# DASHBOARD METRICS
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total News Articles",
        len(df)
    )

with col2:
    st.metric(
        "News Categories",
        df["category"].nunique()
    )

with col3:
    st.metric(
        "AI Clusters",
        df["cluster"].nunique()
    )

with col4:
    st.metric(
        "Silhouette Score",
        f"{silhouette_score_value:.3f}"
    )
st.metric(
    "Classification Accuracy",
    f"{classifier_accuracy * 100:.2f}%"
)

# --------------------------------------------------
# CATEGORY DISTRIBUTION
# --------------------------------------------------

st.subheader("📊 Category Distribution")

category_counts = df["category"].value_counts()

st.bar_chart(category_counts)


# --------------------------------------------------
# EXPLORE NEWS ARTICLES
# --------------------------------------------------

st.subheader("📰 Explore News Articles")

selected_category = st.selectbox(
    "Select News Category",
    sorted(df["category"].unique())
)

filtered_df = df[
    df["category"] == selected_category
]

st.write(
    f"Total articles in {selected_category.title()}: "
    f"{len(filtered_df)}"
)

for article in filtered_df["text"].head(5):

    with st.expander(
        article[:100] + "..."
    ):
        st.write(article)

st.write("---")
st.subheader(f"☁️ {selected_category.title()} Word Cloud")

if not filtered_df.empty:
    text_corpus = " ".join(filtered_df["text"].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='#0f172a', colormap='cool').generate(text_corpus)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    fig.patch.set_facecolor('#0f172a')
    
    st.pyplot(fig)
else:
    st.info("No articles found for the selected category.")


# --------------------------------------------------
# AI GENERATED NEWS CLUSTERS
# --------------------------------------------------

st.subheader("🧠 AI-Generated News Clusters")

cluster_counts = (
    df["cluster"]
    .value_counts()
    .sort_index()
)

st.bar_chart(cluster_counts)


cluster_options = sorted(
    df["cluster"].unique()
)

selected_cluster = st.selectbox(
    "Select AI Cluster",
    cluster_options,
    format_func=lambda x:
    f"Cluster {x} - {topic_names[x]}"
)


cluster_articles = df[
    df["cluster"] == selected_cluster
]


st.write(
    f"### Topic: {topic_names[selected_cluster]}"
)

st.write(
    f"Articles in this cluster: "
    f"{len(cluster_articles)}"
)


# --------------------------------------------------
# CLUSTER KEYWORDS
# --------------------------------------------------

st.write("### 🔑 Top Keywords")

st.write(
    ", ".join(
        cluster_keywords[selected_cluster]
    )
)


# --------------------------------------------------
# CLUSTER ARTICLES
# --------------------------------------------------

st.write("### 📰 Articles in Selected Cluster")

for article in cluster_articles["text"].head(5):

    with st.expander(
        article[:100] + "..."
    ):
        st.write(article)


# --------------------------------------------------
# DISCOVERED TOPICS
# --------------------------------------------------

st.subheader("🔍 Discovered Topics")

for cluster_id in sorted(topic_names):

    st.write(
        f"**Cluster {cluster_id} — "
        f"{topic_names[cluster_id]}:** "
        f"{', '.join(cluster_keywords[cluster_id])}"
    )


# --------------------------------------------------
# SEARCH NEWS ARTICLES
# --------------------------------------------------

st.subheader("🔎 Search News Articles")

search_query = st.text_input(
    "Enter a keyword to search news"
)

if search_query:

    search_results = df[
        df["text"].str.contains(
            search_query,
            case=False,
            na=False
        )
    ]

    st.write(
        f"Found {len(search_results)} matching articles"
    )

    for _, article in search_results.head(10).iterrows():

        with st.expander(
            f"{article['category'].title()} — "
            f"{article['text'][:100]}..."
        ):

            st.write(
                article["text"]
            )


# --------------------------------------------------
# SIMILAR NEWS RECOMMENDATION
# --------------------------------------------------

st.subheader("🔗 Find Similar News Articles")

article_index = st.number_input(
    "Enter Article Index",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)


if st.button("Find Similar Articles"):

    selected_index = int(
        article_index
    )

    st.write(
        "### Selected Article"
    )

    st.write(
        df.iloc[selected_index]["text"]
    )


    similar_indices = find_similar_articles(
        df["text"],
        selected_index
    )


    st.write(
        "### Recommended Similar Articles"
    )


    for index in similar_indices:

        article = df.iloc[index]

        with st.expander(
            f"{article['category'].title()} — "
            f"{article['text'][:100]}..."
        ):

            st.write(
                article["text"]
            )
            # --------------------------------------------------
# CLUSTER VISUALIZATION
# --------------------------------------------------

st.subheader("📈 AI Cluster Visualization")

visualization_df = create_cluster_visualization(
    df["text"],
    df["cluster"]
)

st.scatter_chart(
    visualization_df,
    x="PCA 1",
    y="PCA 2",
    color="Cluster"
)
# --------------------------------------------------
# NEWS CATEGORY PREDICTION
# --------------------------------------------------

st.subheader("🤖 AI News Category Predictor")

news_input = st.text_area(
    "Paste a news article or paragraph",
    height=180
)

if st.button("Predict News Category"):

    if news_input.strip():

        prediction, confidence = predict_category(
            news_input,
            predictor_vectorizer,
            predictor_model
        )

        st.success(
            f"Predicted Category: {prediction.title()}"
        )

        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

    else:
        st.warning("Please enter some news text.")