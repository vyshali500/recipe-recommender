import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Recipe Recommender",
    page_icon="🍲",
    layout="wide"
)

# --------------------------------
# 2. PAGE DESIGN
# --------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #eff8ef;
}

h1, h2, h3, p, label {
    color: black !important;
}

.stMarkdown {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# 3. LOAD DATASET
# --------------------------------

@st.cache_data
def load_data():

    data = pd.read_csv("recipes.csv")

    data.columns = data.columns.str.strip()

    required_columns = [
        "Recipe_Name",
        "Ingredients",
        "Total_Time_Min",
        "Instructions"
    ]

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    data = data.dropna(
        subset=["Recipe_Name", "Ingredients"]
    ).copy()

    data["Ingredients"] = (
        data["Ingredients"].astype(str)
    )

    return data


data = load_data()

# --------------------------------
# 4. DATA PREPROCESSING
# --------------------------------

def clean_ingredients(ingredients):

    return " ".join(
        item.strip().lower()
        for item in str(ingredients).split(";")
        if item.strip()
    )


data["Clean_Ingredients"] = (
    data["Ingredients"].apply(clean_ingredients)
)

# --------------------------------
# 5. TF-IDF VECTORIZATION
# --------------------------------

@st.cache_resource
def build_tfidf_model(ingredient_text):

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(
        ingredient_text
    )

    return vectorizer, tfidf_matrix


vectorizer, tfidf_matrix = build_tfidf_model(
    data["Clean_Ingredients"].tolist()
)

# --------------------------------
# 6. WEBSITE HEADING
# --------------------------------

st.title("🍲 Recipe Recommender System")

st.write(
    "Find delicious recipes using the "
    "ingredients you already have!"
)

st.divider()

# --------------------------------
# 7. INGREDIENT SELECTION
# --------------------------------

st.header("🥕 Choose Your Ingredients")

all_ingredients = []

for items in data["Ingredients"].dropna():

    all_ingredients.extend(
        item.strip()
        for item in items.split(";")
        if item.strip()
    )

all_ingredients = sorted(
    set(all_ingredients),
    key=str.lower
)

selected_ingredients = st.multiselect(
    "Select your available ingredients:",
    all_ingredients
)

# --------------------------------
# 8. RECIPE RECOMMENDATION
# --------------------------------

if st.button("🔍 Find Recipes"):

    if not selected_ingredients:

        st.warning(
            "Please select at least one ingredient."
        )

    else:

        # Convert user ingredients into text
        user_input = " ".join(
            item.lower()
            for item in selected_ingredients
        )

        # Convert user input using TF-IDF
        user_vector = vectorizer.transform(
            [user_input]
        )

        # Calculate Cosine Similarity
        similarity_scores = cosine_similarity(
            user_vector,
            tfidf_matrix
        ).flatten()

        # Store similarity scores
        results = data.copy()

        results["Similarity_Score"] = (
            similarity_scores * 100
        )

        # Sort recipes by similarity
        results = results.sort_values(
            by="Similarity_Score",
            ascending=False
        )

        # Display recommendations
        st.header("✨ Recommended Recipes")

        recommended = results[
            results["Similarity_Score"] > 0
        ].head(20)

        if recommended.empty:

            st.info(
                "No matching recipes found. "
                "Try selecting different ingredients."
            )

        else:

            for _, recipe in recommended.iterrows():

                percentage = recipe[
                    "Similarity_Score"
                ]

                with st.expander(
                    f"🍽️ {recipe['Recipe_Name']} "
                    f"- {percentage:.2f}% Similarity"
                ):

                    st.write("🥗 **Ingredients:**")

                    st.write(
                        recipe["Ingredients"]
                    )

                    st.write("⏱️ **Preparation Time:**")

                    st.write(
                        recipe["Total_Time_Min"],
                        "minutes"
                    )

                    st.write("👩‍🍳 **Instructions:**")

                    st.write(
                        recipe["Instructions"]
                    )

st.divider()

st.caption(
    "Made with 💚 Recipe Recommender System"
)
