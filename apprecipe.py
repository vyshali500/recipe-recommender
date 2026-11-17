import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    silhouette_score
)
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


# =====================================
# 1. PAGE SETTINGS
# =====================================

st.set_page_config(
    page_title="AI Recipe Recommender",
    page_icon="🍲",
    layout="wide"
)


# =====================================
# 2. WEBSITE DESIGN
# =====================================

st.markdown("""
<style>

.stApp {
    background-color: #f1f8e9 !important;
    color: #172b17 !important;
}

.stApp p,
.stApp label,
.stApp li,
.stApp span,
.stApp small,
.stApp div[data-testid="stMarkdownContainer"] {
    color: #172b17 !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #14532d !important;
}

.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background-color: white !important;
    color: #172b17 !important;
}

[data-baseweb="select"] {
    background-color: white !important;
    color: #172b17 !important;
}

[data-baseweb="select"] * {
    color: #172b17 !important;
}

[role="listbox"],
[role="option"] {
    background-color: white !important;
    color: #172b17 !important;
}

.stButton button {
    background-color: #2e7d32 !important;
    color: white !important;
    border-radius: 10px;
    font-weight: bold;
}

.stButton button p {
    color: white !important;
}

button[data-baseweb="tab"] {
    color: #14532d !important;
}

[data-testid="stExpander"] {
    background-color: white !important;
    border: 1px solid #a5d6a7 !important;
    border-radius: 10px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span {
    color: #14532d !important;
}

[data-testid="stMetric"] {
    background-color: white !important;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #a5d6a7;
}

[data-testid="stMetricValue"],
[data-testid="stMetricLabel"] {
    color: #14532d !important;
}

[data-testid="stSidebar"] {
    background-color: #e0f2df !important;
}

[data-testid="stSidebar"] * {
    color: #172b17 !important;
}

[data-testid="stDataFrame"] {
    background-color: white !important;
}

.stCaption {
    color: #365b36 !important;
}

a {
    color: #1565c0 !important;
}

</style>
""", unsafe_allow_html=True)


# =====================================
# 3. LOAD DATASET
# =====================================

@st.cache_data
def load_data():
    return pd.read_csv("recipes.csv")


try:
    data = load_data()

except Exception as e:
    st.error("Unable to load recipes.csv")
    st.write("Make sure recipes.csv is in the same folder as apprecipe.py.")
    st.code(str(e))
    st.stop()


required_columns = [
    "Recipe_Name",
    "Ingredients",
    "Cuisine",
    "Diet_Type",
    "Difficulty",
    "Prep_Time_Min",
    "Cook_Time_Min",
    "Total_Time_Min",
    "Instructions"
]

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:
    st.error("Missing columns: " + ", ".join(missing_columns))
    st.stop()


data = data.dropna(
    subset=["Recipe_Name", "Ingredients", "Difficulty"]
).copy()

if data.empty:
    st.error("No valid recipes found in the dataset.")
    st.stop()

for col in ["Cuisine", "Diet_Type", "Instructions"]:
    data[col] = data[col].fillna("").astype(str)

for col in ["Prep_Time_Min", "Cook_Time_Min", "Total_Time_Min"]:
    data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)


# =====================================
# 4. FEATURE PREPARATION
# =====================================

def make_features(frame):

    return (
        frame["Ingredients"].fillna("").astype(str) + " " +
        frame["Cuisine"].fillna("").astype(str) + " " +
        frame["Diet_Type"].fillna("").astype(str) + " " +
        frame["Prep_Time_Min"].astype(str) + " " +
        frame["Cook_Time_Min"].astype(str)
    )


# =====================================
# 5. SUPERVISED ML MODEL
# LOGISTIC REGRESSION
# =====================================

@st.cache_resource
def train_supervised_model(frame):

    X = make_features(frame)
    y = frame["Difficulty"].astype(str)

    if len(frame) < 10 or y.nunique() < 2:
        return None

    if y.value_counts().min() < 2:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer()

    X_train_vector = vectorizer.fit_transform(X_train)
    X_test_vector = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_vector, y_train)

    predictions = model.predict(X_test_vector)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions,
            average="weighted",
            zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions,
            average="weighted",
            zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions,
            average="weighted",
            zero_division=0
        ),
        "report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0
        ),
        "matrix": confusion_matrix(
            y_test,
            predictions,
            labels=model.classes_
        ),
        "classes": model.classes_
    }


# =====================================
# 6. UNSUPERVISED ML MODEL
# K-MEANS CLUSTERING
# =====================================

@st.cache_resource
def train_clustering_model(frame):

    ingredients = frame["Ingredients"].fillna("").astype(str)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(ingredients)

    if len(frame) < 6:
        return None

    k = min(5, len(frame) - 1)

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    score = silhouette_score(
        X,
        labels,
        metric="cosine",
        sample_size=min(500, len(frame)),
        random_state=42
    )

    return {
        "model": model,
        "vectorizer": vectorizer,
        "labels": labels,
        "score": score
    }


# =====================================
# 7. TRAIN MODELS
# =====================================

with st.spinner("Training AI models..."):

    supervised_results = train_supervised_model(data)
    clustering_results = train_clustering_model(data)


# =====================================
# 8. HEADER
# =====================================

st.title("🍃 AI RECIPE RECOMMENDER")

st.subheader("Discover delicious recipes using Artificial Intelligence!")

st.write(
    "Choose your available ingredients and find recipes that match them."
)

st.divider()


# =====================================
# 9. NAVIGATION
# =====================================

tab1, tab2, tab3 = st.tabs([
    "🍲 RECIPE FINDER",
    "🤖 MODEL EVALUATION",
    "📊 RECIPE CLUSTERING"
])


# =====================================
# 10. RECIPE FINDER
# =====================================

with tab1:

    st.header("🥕 Find Recipes Using Your Ingredients")

    ingredients_list = []

    for items in data["Ingredients"].dropna():

        ingredients_list.extend([
            item.strip()
            for item in str(items).split(";")
            if item.strip()
        ])

    ingredients_list = sorted(set(ingredients_list))

    selected = st.multiselect(
        "Select your available ingredients",
        ingredients_list
    )

    cuisine_options = ["All"] + sorted(
        data["Cuisine"].unique().tolist()
    )

    diet_options = ["All"] + sorted(
        data["Diet_Type"].unique().tolist()
    )

    cuisine = st.selectbox(
        "🌍 Select Cuisine",
        cuisine_options
    )

    diet = st.selectbox(
        "🥗 Select Diet Type",
        diet_options
    )

    if st.button("🔍 FIND MY RECIPES"):

        if not selected:

            st.warning("Please select at least one ingredient.")

        else:

            recipe_text = data["Ingredients"].fillna("").astype(str)

            vectorizer = TfidfVectorizer(
                token_pattern=r"(?u)\b\w+\b"
            )

            recipe_matrix = vectorizer.fit_transform(recipe_text)

            query_vector = vectorizer.transform(
                [" ".join(selected)]
            )

            similarities = cosine_similarity(
                query_vector,
                recipe_matrix
            ).flatten()

            ranked_indices = np.argsort(similarities)[::-1]

            st.subheader("✨ Recommended Recipes")

            shown = 0

            for index in ranked_indices:

                recipe = data.iloc[index]

                if similarities[index] <= 0:
                    continue

                if cuisine != "All" and recipe["Cuisine"] != cuisine:
                    continue

                if diet != "All" and recipe["Diet_Type"] != diet:
                    continue

                with st.expander(
                    "🍲 " + str(recipe["Recipe_Name"])
                ):

                    st.metric(
                        "Ingredient Similarity",
                        f"{similarities[index] * 100:.2f}%"
                    )

                    st.write("🌍 Cuisine:", recipe["Cuisine"])
                    st.write("🥗 Diet:", recipe["Diet_Type"])
                    st.write("⭐ Difficulty:", recipe["Difficulty"])

                    st.write(
                        "⏱️ Total Time:",
                        recipe["Total_Time_Min"],
                        "minutes"
                    )

                    st.write("🧺 Ingredients:")
                    st.write(recipe["Ingredients"])

                    st.write("👩‍🍳 Instructions:")
                    st.write(recipe["Instructions"])

                shown += 1

                if shown == 10:
                    break

            if shown == 0:
                st.info("No matching recipes found. Try different ingredients.")


# =====================================
# 11. MODEL EVALUATION
# =====================================

with tab2:

    st.header("🤖 Supervised Machine Learning")

    st.write("Algorithm: Logistic Regression")

    st.write(
        "Purpose: Predict the difficulty of a recipe using its ingredients, cuisine, diet and cooking time."
    )

    if supervised_results is None:

        st.warning(
            "Not enough examples in each difficulty class to train the supervised model."
        )

    else:

        result = supervised_results

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Accuracy",
            f"{result['accuracy'] * 100:.2f}%"
        )

        col2.metric(
            "Precision",
            f"{result['precision'] * 100:.2f}%"
        )

        col3.metric(
            "Recall",
            f"{result['recall'] * 100:.2f}%"
        )

        col4.metric(
            "F1 Score",
            f"{result['f1'] * 100:.2f}%"
        )

        st.subheader("📋 Classification Report")

        report_df = pd.DataFrame(
            result["report"]
        ).transpose()

        st.dataframe(
            report_df,
            use_container_width=True
        )

        st.subheader("📊 Confusion Matrix")

        matrix_df = pd.DataFrame(
            result["matrix"],
            index=result["classes"],
            columns=result["classes"]
        )

        st.dataframe(
            matrix_df,
            use_container_width=True
        )

        st.divider()

        st.subheader("🍲 Predict Recipe Difficulty")

        recipe_choice = st.selectbox(
            "Choose a recipe",
            data["Recipe_Name"].astype(str).tolist()
        )

        if st.button("PREDICT DIFFICULTY"):

            selected_recipe = data[
                data["Recipe_Name"].astype(str) == recipe_choice
            ].iloc[[0]]

            features = make_features(selected_recipe)

            prediction = result["model"].predict(
                result["vectorizer"].transform(features)
            )[0]

            st.success(
                "Predicted Recipe Difficulty: " + str(prediction)
            )

    st.divider()

    st.header("📊 Unsupervised Machine Learning")

    st.write("Algorithm: K-Means Clustering")

    st.write(
        "Purpose: Group similar recipes based on their ingredients without using predefined labels."
    )

    if clustering_results is None:

        st.warning("Not enough recipes for clustering.")

    else:

        st.metric(
            "Silhouette Score",
            f"{clustering_results['score']:.4f}"
        )

        st.metric(
            "Number of Clusters",
            clustering_results["model"].n_clusters
        )

        st.write(
            "A higher silhouette score generally indicates better-separated clusters."
        )


# =====================================
# 12. RECIPE CLUSTERING
# =====================================

with tab3:

    st.header("📊 Similar Recipe Groups")

    if clustering_results is None:

        st.warning("Not enough recipes for clustering.")

    else:

        clustered_data = data.copy()

        clustered_data["Cluster"] = clustering_results["labels"]

        selected_cluster = st.selectbox(
            "Select a recipe group",
            sorted(clustered_data["Cluster"].unique()),
            key="cluster_select"
        )

        group = clustered_data[
            clustered_data["Cluster"] == selected_cluster
        ]

        st.metric(
            "Recipes in this group",
            len(group)
        )

        st.dataframe(
            group[
                [
                    "Recipe_Name",
                    "Ingredients",
                    "Cuisine",
                    "Difficulty"
                ]
            ],
            use_container_width=True
        )


# =====================================
# 13. FOOTER
# =====================================

st.divider()

st.markdown(
    "<center>🍃 AI Recipe Recommender | Powered by Machine Learning</center>",
    unsafe_allow_html=True
)
