# ============================================================
# MovieIQ - Predictive Analytics on Film Success
# ============================================================

import ast

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind, chi2_contingency

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MovieIQ",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("MovieIQ")
st.subheader("Predictive Analytics on Film Success")

st.write(
    "MovieIQ analyzes movie data and predicts whether a movie is "
    "likely to be financially successful based on its available features."
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    # Load the movie dataset
    df = pd.read_csv("data/movies.csv")

    return df


df = load_data()


# ============================================================
# DATA PREPARATION
# ============================================================

@st.cache_data
def prepare_data(df):

    df = df.copy()

    # Remove movies where budget or revenue is zero
    # because these values cannot be reliably used to determine success
    df = df[
        (df["budget"] > 0) &
        (df["revenue"] > 0)
    ].copy()

    # Create the target variable
    # 1 = Successful
    # 0 = Unsuccessful
    df["success"] = np.where(
        df["revenue"] > df["budget"],
        1,
        0
    )

    # Function to process the genres column
    def process_genres(x):

        # Handle missing values
        if x is None or (
            isinstance(x, float) and pd.isna(x)
        ):
            return ["Unknown"]

        # Handle strings
        if isinstance(x, str):

            try:
                x = ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return [x]

        # Handle lists
        if isinstance(x, list):

            genres = []

            for genre in x:

                if isinstance(genre, dict):
                    genres.append(genre["name"])

                elif isinstance(genre, str):
                    genres.append(genre)

            return genres if genres else ["Unknown"]

        return ["Unknown"]

    # Process genres
    df["genres"] = df["genres"].apply(process_genres)

    return df


df = prepare_data(df)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Movie Filters")


# Get all available genres
all_genres = sorted(
    set(
        genre
        for genre_list in df["genres"]
        for genre in genre_list
    )
)


# Genre filter
selected_genre = st.sidebar.selectbox(
    "Select Genre",
    ["All Genres"] + all_genres
)


# Minimum vote average filter
min_vote = st.sidebar.slider(
    "Minimum Vote Average",
    min_value=float(df["vote_average"].min()),
    max_value=float(df["vote_average"].max()),
    value=float(df["vote_average"].min()),
    step=0.1
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df[
    df["vote_average"] >= min_vote
].copy()


# Apply genre filter if a specific genre is selected
if selected_genre != "All Genres":

    filtered_df = filtered_df[
        filtered_df["genres"].apply(
            lambda genres: selected_genre in genres
        )
    ]


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.header("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Movies",
        len(filtered_df)
    )

with col2:
    st.metric(
        "Successful Movies",
        int(filtered_df["success"].sum())
    )

with col3:
    success_rate = (
        filtered_df["success"].mean() * 100
        if len(filtered_df) > 0
        else 0
    )

    st.metric(
        "Success Rate",
        f"{success_rate:.1f}%"
    )

with col4:
    st.metric(
        "Average Vote",
        f"{filtered_df['vote_average'].mean():.2f}"
        if len(filtered_df) > 0
        else "0.00"
    )


# ============================================================
# EDA SECTION
# ============================================================

st.header("Exploratory Data Analysis")


if len(filtered_df) == 0:

    st.warning(
        "No movies match the selected filters. "
        "Try changing the filters."
    )

else:

    # --------------------------------------------------------
    # Budget vs Revenue
    # --------------------------------------------------------

    st.subheader("Budget vs Revenue")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.scatterplot(
        data=filtered_df,
        x="budget",
        y="revenue",
        hue="success",
        ax=ax
    )

    ax.set_title("Budget vs Revenue")
    ax.set_xlabel("Budget")
    ax.set_ylabel("Revenue")

    st.pyplot(fig)

    st.write(
        "This chart shows the relationship between a movie's budget "
        "and its revenue. The filters in the sidebar update the chart."
    )


    # --------------------------------------------------------
    # Genre Success Rate
    # --------------------------------------------------------

    st.subheader("Genre Success Rate")

    genre_data = filtered_df.explode("genres")

    genre_success = (
        genre_data
        .groupby("genres")["success"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        * 100
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        x=genre_success.values,
        y=genre_success.index,
        ax=ax
    )

    ax.set_title("Success Rate by Genre")
    ax.set_xlabel("Success Rate (%)")
    ax.set_ylabel("Genre")

    st.pyplot(fig)


    # --------------------------------------------------------
    # Popularity vs Success
    # --------------------------------------------------------

    st.subheader("Popularity vs Success")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(
        data=filtered_df,
        x="success",
        y="popularity",
        ax=ax
    )

    ax.set_title("Popularity vs Movie Success")
    ax.set_xlabel("Success (0 = Unsuccessful, 1 = Successful)")
    ax.set_ylabel("Popularity")

    st.pyplot(fig)


    # --------------------------------------------------------
    # Runtime vs Success
    # --------------------------------------------------------

    st.subheader("Runtime vs Success")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(
        data=filtered_df,
        x="success",
        y="runtime",
        ax=ax
    )

    ax.set_title("Runtime vs Movie Success")
    ax.set_xlabel("Success (0 = Unsuccessful, 1 = Successful)")
    ax.set_ylabel("Runtime (minutes)")

    st.pyplot(fig)


    # --------------------------------------------------------
    # Vote Average vs Success
    # --------------------------------------------------------

    st.subheader("Vote Average vs Success")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(
        data=filtered_df,
        x="success",
        y="vote_average",
        ax=ax
    )

    ax.set_title("Vote Average vs Movie Success")
    ax.set_xlabel("Success (0 = Unsuccessful, 1 = Successful)")
    ax.set_ylabel("Vote Average")

    st.pyplot(fig)


    # --------------------------------------------------------
    # Correlation Heatmap
    # --------------------------------------------------------

    st.subheader("Correlation Heatmap")

    numeric_columns = [
        "budget",
        "revenue",
        "popularity",
        "runtime",
        "vote_average",
        "success"
    ]

    correlation = filtered_df[numeric_columns].corr()

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title("Correlation Heatmap")

    st.pyplot(fig)


# ============================================================
# STATISTICAL TESTING
# ============================================================

st.header("Statistical Testing")


# ------------------------------------------------------------
# T-TEST
# ------------------------------------------------------------

st.subheader("T-Test: Popularity vs Success")

successful_popularity = filtered_df[
    filtered_df["success"] == 1
]["popularity"]

unsuccessful_popularity = filtered_df[
    filtered_df["success"] == 0
]["popularity"]


if len(successful_popularity) > 1 and len(unsuccessful_popularity) > 1:

    t_stat, t_p_value = ttest_ind(
        successful_popularity,
        unsuccessful_popularity,
        equal_var=False
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "T-Statistic",
            f"{t_stat:.4f}"
        )

    with col2:
        st.metric(
            "P-Value",
            f"{t_p_value:.6f}"
        )

    if t_p_value < 0.05:

        st.success(
            "The p-value is below 0.05, so we reject the null "
            "hypothesis. There is a statistically significant "
            "difference in popularity between successful and "
            "unsuccessful movies."
        )

    else:

        st.info(
            "The p-value is greater than or equal to 0.05, so "
            "we fail to reject the null hypothesis."
        )


# ------------------------------------------------------------
# CHI-SQUARE TEST
# ------------------------------------------------------------

st.subheader("Chi-Square Test: Genre vs Success")

genre_test_df = filtered_df.explode("genres")

genre_success_table = pd.crosstab(
    genre_test_df["genres"],
    genre_test_df["success"]
)


# Make sure both success categories exist
if 0 not in genre_success_table.columns:

    genre_success_table[0] = 0

if 1 not in genre_success_table.columns:

    genre_success_table[1] = 0


genre_success_table = genre_success_table[
    [0, 1]
]

genre_success_table.columns = [
    "Unsuccessful",
    "Successful"
]


if genre_success_table.shape[0] > 1:

    chi2_stat, chi2_p_value, degrees_freedom, expected = (
        chi2_contingency(genre_success_table)
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Chi-Square Statistic",
            f"{chi2_stat:.4f}"
        )

    with col2:
        st.metric(
            "Degrees of Freedom",
            degrees_freedom
        )

    with col3:
        st.metric(
            "P-Value",
            f"{chi2_p_value:.6f}"
        )

    if chi2_p_value < 0.05:

        st.success(
            "The p-value is below 0.05, so we reject the null "
            "hypothesis. Genre and movie success have a "
            "statistically significant association."
        )

    else:

        st.info(
            "The p-value is greater than or equal to 0.05, so "
            "we fail to reject the null hypothesis."
        )


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

st.header("Movie Success Predictor")


# ------------------------------------------------------------
# Prepare features
# ------------------------------------------------------------

# Create a copy for machine learning
model_df = df.copy()


# Create genre dummy variables
genre_dummies = model_df["genres"].explode().str.get_dummies()

# Combine multiple genres belonging to the same movie
genre_dummies = genre_dummies.groupby(
    genre_dummies.index
).max()


# Numeric features used by the model
numeric_features = [
    "budget",
    "popularity",
    "runtime",
    "vote_average"
]


# Combine numeric features and genre features
X = pd.concat(
    [
        model_df[numeric_features],
        genre_dummies
    ],
    axis=1
)


# Target variable
y = model_df["success"]


# ------------------------------------------------------------
# Train-test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# Train Random Forest
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# Model evaluation
# ------------------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)


# Display model metrics
st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Accuracy",
        f"{accuracy:.2%}"
    )

with col2:
    st.metric(
        "Precision",
        f"{precision:.2%}"
    )

with col3:
    st.metric(
        "Recall",
        f"{recall:.2%}"
    )


# ============================================================
# MOVIE PREDICTION FORM
# ============================================================

st.subheader("Predict Movie Success")

st.write(
    "Enter the movie details below to get a prediction from "
    "the Random Forest model."
)


col1, col2 = st.columns(2)


with col1:

    input_budget = st.number_input(
        "Budget",
        min_value=0.0,
        value=float(df["budget"].median())
    )

    input_popularity = st.number_input(
        "Popularity",
        min_value=0.0,
        value=float(df["popularity"].median())
    )


with col2:

    input_runtime = st.number_input(
        "Runtime (minutes)",
        min_value=0.0,
        value=float(df["runtime"].median())
    )

    input_vote_average = st.number_input(
        "Vote Average",
        min_value=0.0,
        max_value=10.0,
        value=float(df["vote_average"].median())
    )


input_genre = st.selectbox(
    "Movie Genre",
    all_genres
)


# ------------------------------------------------------------
# Create input row
# ------------------------------------------------------------

input_data = pd.DataFrame(
    0,
    index=[0],
    columns=X.columns
)


# Add numeric values
input_data["budget"] = input_budget
input_data["popularity"] = input_popularity
input_data["runtime"] = input_runtime
input_data["vote_average"] = input_vote_average


# Add selected genre
if input_genre in input_data.columns:
    input_data[input_genre] = 1


# ------------------------------------------------------------
# Prediction button
# ------------------------------------------------------------

if st.button("Predict Movie Success"):

    prediction = model.predict(
        input_data
    )[0]

    probability = model.predict_proba(
        input_data
    )[0][1]


    if prediction == 1:

        st.success(
            f"Prediction: Successful Movie"
        )

    else:

        st.error(
            f"Prediction: Not Successful"
        )

    st.write(
        f"Estimated probability of success: "
        f"{probability:.2%}"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MovieIQ | Predictive Analytics on Film Success"
)