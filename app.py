
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from collections import Counter

from modules.preprocessing import preprocess_documents
from modules.corpus_analytics import (
    get_vocabulary,
    get_top_terms,
    get_document_lengths
)
from modules.duplicates_detection import (
    exact_duplicates,
    near_duplicates
)


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Text Mining & Document Intelligence",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Text Mining & Document Intelligence")
st.markdown(
    "An interactive system for corpus analytics, "
    "text preprocessing, duplicate detection and vocabulary analysis."
)

st.divider()


# --------------------------------------------------
# 2. PATH CONFIGURATION
# --------------------------------------------------

DATA_PATH = Path("data/raw_dataset.csv")

EXPERIMENT_PATH = Path(
    "results/preprocessing_experiments"
)


# --------------------------------------------------
# 3. LOAD DATASET
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a Module",
    [
        "Dataset Overview",
        "Text Preprocessing",
        "Experiment Comparison",
        "Duplicate Analysis"
    ]
)


def load_data():
    """Load the default dataset or an uploaded CSV."""

    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    uploaded_file = st.sidebar.file_uploader(
        "Upload your dataset",
        type=["csv"]
    )

    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    return None


df = load_data()

if df is None:
    st.warning(
        "Dataset not found. Please upload a CSV file "
        "or place raw_dataset.csv inside the data folder."
    )
    st.stop()


# --------------------------------------------------
# 4. DATA VALIDATION
# --------------------------------------------------

required_columns = {
    "document_id",
    "title",
    "text",
    "category"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    st.error(
        f"Dataset is missing required columns: "
        f"{sorted(missing_columns)}"
    )
    st.stop()


# Remove rows without usable document text or ID.
df = df.copy()

df["document_id"] = df["document_id"].fillna("").astype(str)
df["title"] = df["title"].fillna("").astype(str)
df["text"] = df["text"].fillna("").astype(str)
df["category"] = df["category"].fillna("Unknown").astype(str)

df = df[
    (df["document_id"].str.strip() != "")
    & (
        df["title"].str.strip().ne("")
        | df["text"].str.strip().ne("")
    )
].copy()

if df.empty:
    st.error("No valid documents found in the dataset.")
    st.stop()


documents = (
    df["title"] + " " + df["text"]
).tolist()


# --------------------------------------------------
# 5. HELPER FUNCTIONS
# --------------------------------------------------

def get_tokens_statistics(token_documents):
    """Calculate token and vocabulary statistics."""

    all_tokens = [
        token
        for document in token_documents
        for token in document
    ]

    lengths = [
        len(document)
        for document in token_documents
    ]

    return {
        "total_tokens": len(all_tokens),
        "vocabulary_size": len(set(all_tokens)),
        "average_length": (
            sum(lengths) / len(lengths) if lengths else 0
        ),
        "minimum_length": min(lengths) if lengths else 0,
        "maximum_length": max(lengths) if lengths else 0
    }


def create_top_terms_dataframe(token_documents, top_n=20):
    """Convert top frequent terms into a DataFrame."""

    top_terms = get_top_terms(
        token_documents,
        top_n=top_n
    )

    if isinstance(top_terms, dict):
        return pd.DataFrame(
            top_terms.items(),
            columns=["Word", "Frequency"]
        )

    return pd.DataFrame(
        top_terms,
        columns=["Word", "Frequency"]
    )


def display_metric_row(statistics):
    """Display token statistics as metric cards."""

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Tokens",
        f"{statistics['total_tokens']:,}"
    )

    col2.metric(
        "Vocabulary Size",
        f"{statistics['vocabulary_size']:,}"
    )

    col3.metric(
        "Average Tokens / Document",
        f"{statistics['average_length']:.2f}"
    )


# --------------------------------------------------
# 6. DATASET OVERVIEW
# --------------------------------------------------

if page == "Dataset Overview":

    st.header("Dataset Overview")

    st.write(
        "Explore the BBC News dataset and understand "
        "its structure and distribution."
    )

    # Main metrics
    total_words = sum(
        len(document.split())
        for document in documents
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Documents",
        f"{len(df):,}"
    )

    col2.metric(
        "Categories",
        df["category"].nunique()
    )

    col3.metric(
        "Unique Document IDs",
        df["document_id"].nunique()
    )

    col4.metric(
        "Total Words",
        f"{total_words:,}"
    )

    st.divider()

    # Category distribution
    st.subheader("Category Distribution")

    category_counts = df["category"].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))

        category_counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel("Category")
        ax.set_ylabel("Number of Documents")
        ax.set_title("Documents per Category")
        ax.tick_params(axis="x", rotation=30)

        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))

        category_counts.plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=ax
        )

        ax.set_ylabel("")
        ax.set_title("Category Percentage")

        st.pyplot(fig)
        plt.close(fig)

    # Category-wise statistics
    st.subheader("Category-wise Statistics")

    category_stats = (
        df.groupby("category")
        .agg(
            document_count=("document_id", "count"),
            average_text_length=(
                "text",
                lambda x: x.str.split().str.len().mean()
            )
        )
        .reset_index()
    )

    category_stats["percentage"] = (
        category_stats["document_count"]
        / len(df) * 100
    ).round(2)

    category_stats["average_text_length"] = (
        category_stats["average_text_length"].round(2)
    )

    st.dataframe(
        category_stats,
        use_container_width=True
    )

    st.divider()

    # Document length distribution
    st.subheader("Document Length Distribution")

    document_lengths = [
        len(document.split())
        for document in documents
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        document_lengths,
        bins=25
    )

    ax.set_xlabel("Words per Document")
    ax.set_ylabel("Number of Documents")
    ax.set_title("Distribution of Document Lengths")

    st.pyplot(fig)
    plt.close(fig)

    # Vocabulary analysis
    st.divider()

    st.subheader("Vocabulary Analysis")

    with st.spinner("Calculating vocabulary statistics..."):

        overview_tokens = preprocess_documents(
            documents,
            lowercase=True,
            remove_stopword=True
        )

        vocabulary_stats = get_tokens_statistics(
            overview_tokens
        )

        top_terms_df = create_top_terms_dataframe(
            overview_tokens,
            top_n=20
        )

    display_metric_row(vocabulary_stats)

    st.write("Top 20 Most Frequent Words")

    if not top_terms_df.empty:

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(
            top_terms_df["Word"],
            top_terms_df["Frequency"]
        )

        ax.invert_yaxis()
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Word")
        ax.set_title("Top 20 Frequent Words")

        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            top_terms_df,
            use_container_width=True
        )

    # Document preview
    st.divider()

    st.subheader("Document Preview")

    selected_category = st.selectbox(
        "Filter by Category",
        ["All"] + sorted(df["category"].unique().tolist())
    )

    if selected_category != "All":
        preview_df = df[
            df["category"] == selected_category
        ]
    else:
        preview_df = df

    st.dataframe(
        preview_df[
            ["document_id", "title", "category", "text"]
        ].head(10),
        use_container_width=True
    )


# --------------------------------------------------
# 7. TEXT PREPROCESSING
# --------------------------------------------------

elif page == "Text Preprocessing":

    st.header("Configurable Text Preprocessing")

    st.write(
        "Apply different preprocessing techniques "
        "to the BBC News documents."
    )

    st.subheader("Select Preprocessing Options")

    remove_stopwords = st.checkbox(
        "Remove Stop Words",
        value=True
    )

    normalization_method = st.selectbox(
        "Word Normalization",
        [
            "None",
            "Stemming",
            "Lemmatization"
        ]
    )

    ngram_size = st.selectbox(
        "N-gram Size",
        [1, 2, 3]
    )

    if st.button(
        "Run Preprocessing",
        type="primary"
    ):

        stemming = normalization_method == "Stemming"

        lemmatization = (
            normalization_method == "Lemmatization"
        )

        try:

            with st.spinner("Processing documents..."):

                tokens = preprocess_documents(
                    documents,
                    lowercase=True,
                    remove_stopword=remove_stopwords,
                    stemming=stemming,
                    lemmatization=lemmatization
                )

            output_df = df.copy()

            output_df["tokens"] = [
                json.dumps(document_tokens)
                for document_tokens in tokens
            ]

            output_df["processed_text"] = [
                " ".join(document_tokens)
                for document_tokens in tokens
            ]

            st.session_state["processed_df"] = output_df
            st.session_state["processed_tokens"] = tokens

            st.success(
                "Text preprocessing completed successfully!"
            )

            statistics = get_tokens_statistics(tokens)

            display_metric_row(statistics)

            # N-gram generation
            if ngram_size > 1:

                ngrams = [
                    tuple(
                        document_tokens[i:i + ngram_size]
                    )
                    for document_tokens in tokens
                    for i in range(
                        len(document_tokens) - ngram_size + 1
                    )
                ]

                st.subheader(
                    f"Sample {ngram_size}-grams"
                )

                st.write(
                    f"Total generated {ngram_size}-grams: "
                    f"{len(ngrams):,}"
                )

                st.write(ngrams[:20])

            # Top frequent terms
            st.subheader("Top Frequent Terms")

            top_terms_df = create_top_terms_dataframe(
                tokens,
                top_n=20
            )

            if not top_terms_df.empty:

                fig, ax = plt.subplots(figsize=(10, 5))

                ax.barh(
                    top_terms_df["Word"],
                    top_terms_df["Frequency"]
                )

                ax.invert_yaxis()
                ax.set_xlabel("Frequency")

                st.pyplot(fig)
                plt.close(fig)

            st.subheader("Processed Dataset Preview")

            st.dataframe(
                output_df.head(10),
                use_container_width=True
            )

            # Download
            csv = output_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Processed Dataset",
                data=csv,
                file_name="processed_dataset.csv",
                mime="text/csv"
            )

        except LookupError as error:
            st.error(str(error))

        except Exception as error:
            st.error(
                f"Preprocessing failed: {error}"
            )

    # Show previously processed data
    if "processed_df" in st.session_state:

        st.divider()

        st.subheader("Previously Processed Dataset")

        st.dataframe(
            st.session_state["processed_df"].head(10),
            use_container_width=True
        )

        csv = st.session_state[
            "processed_df"
        ].to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Latest Processed Dataset",
            data=csv,
            file_name="processed_dataset.csv",
            mime="text/csv",
            key="download_latest"
        )


# --------------------------------------------------
# 8. EXPERIMENT COMPARISON
# --------------------------------------------------

elif page == "Experiment Comparison":

    st.header("Preprocessing Experiment Comparison")

    st.write(
        "Compare the impact of different preprocessing "
        "techniques on vocabulary and token statistics."
    )

    comparison_file = (
        EXPERIMENT_PATH / "experiment_comparison.csv"
    )

    if comparison_file.exists():

        results = pd.read_csv(comparison_file)

        st.subheader("Experiment Results")

        st.dataframe(
            results,
            use_container_width=True
        )

        # Vocabulary comparison
        st.subheader("Vocabulary Size Comparison")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(
            results["experiment"],
            results["vocabulary_size"]
        )

        ax.set_xlabel("Experiment")
        ax.set_ylabel("Vocabulary Size")
        ax.set_title("Vocabulary Size Across Experiments")
        ax.tick_params(axis="x", rotation=25)

        st.pyplot(fig)
        plt.close(fig)

        # Total token comparison
        st.subheader("Total Token Comparison")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(
            results["experiment"],
            results["total_tokens"]
        )

        ax.set_xlabel("Experiment")
        ax.set_ylabel("Total Tokens")
        ax.set_title("Total Tokens Across Experiments")
        ax.tick_params(axis="x", rotation=25)

        st.pyplot(fig)
        plt.close(fig)

        # Average document length comparison
        st.subheader("Average Document Length Comparison")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(
            results["experiment"],
            results["average_tokens_per_document"]
        )

        ax.set_xlabel("Experiment")
        ax.set_ylabel("Average Tokens per Document")
        ax.set_title("Average Document Length")
        ax.tick_params(axis="x", rotation=25)

        st.pyplot(fig)
        plt.close(fig)

        # Download comparison
        csv = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Experiment Comparison",
            data=csv,
            file_name="experiment_comparison.csv",
            mime="text/csv"
        )

        # Experiment interpretation
        st.divider()

        st.subheader("Experiment Interpretation")

        baseline = results[
            results["experiment"] == "baseline"
        ]

        if not baseline.empty:

            baseline_vocab = int(
                baseline.iloc[0]["vocabulary_size"]
            )

            for _, row in results.iterrows():

                if row["experiment"] == "baseline":
                    continue

                difference = (
                    baseline_vocab
                    - int(row["vocabulary_size"])
                )

                st.write(
                    f"**{row['experiment']}**: "
                    f"Vocabulary changed by {difference:+,} "
                    f"terms compared with the baseline."
                )

    else:

        st.warning(
            "Experiment results not found. "
            "Run run_preprocessing_experiments.py first."
        )


# --------------------------------------------------
# 9. DUPLICATE ANALYSIS
# --------------------------------------------------

elif page == "Duplicate Analysis":

    st.header("Duplicate Document Detection")

    st.write(
        "Identify exact and near-duplicate documents "
        "in the corpus."
    )

    threshold = st.slider(
        "Near-Duplicate Similarity Threshold",
        min_value=0.50,
        max_value=1.00,
        value=0.80,
        step=0.05
    )

    st.info(
        "Exact duplicates are detected using normalized text. "
        "Near duplicates use Jaccard similarity over token sets."
    )

    if st.button(
        "Run Duplicate Detection",
        type="primary"
    ):

        with st.spinner(
            "Comparing documents. This may take a while..."
        ):

            raw_documents = (
                df["text"].fillna("").astype(str).tolist()
            )

            exact = exact_duplicates(
                raw_documents
            )

            token_documents = preprocess_documents(
                documents,
                lowercase=True,
                remove_stopword=False
            )

            near = near_duplicates(
                token_documents,
                threshold=threshold
            )

        st.session_state["exact_duplicates"] = exact
        st.session_state["near_duplicates"] = near
        st.session_state["duplicate_threshold"] = threshold

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Documents",
            len(df)
        )

        col2.metric(
            "Exact Duplicate Groups",
            len(exact)
        )

        col3.metric(
            "Near Duplicate Pairs",
            len(near)
        )

    if "exact_duplicates" in st.session_state:

        st.divider()

        st.subheader("Exact Duplicate Results")

        exact = st.session_state["exact_duplicates"]

        if exact:

            st.write(
                "The following groups contain documents "
                "with identical normalized text."
            )

            for group_number, group in enumerate(
                exact,
                start=1
            ):

                with st.expander(
                    f"Duplicate Group {group_number}"
                ):

                    for index in group:

                        st.write(
                            f"**Document ID:** "
                            f"{df.iloc[index]['document_id']}"
                        )

                        st.write(
                            f"**Title:** "
                            f"{df.iloc[index]['title']}"
                        )

        else:

            st.success(
                "No exact duplicate groups were found."
            )

        st.divider()

        st.subheader("Near Duplicate Results")

        near = st.session_state["near_duplicates"]

        if near:

            near_df = pd.DataFrame(near)

            st.dataframe(
                near_df,
                use_container_width=True
            )

            st.caption(
                "Inspect the returned similarity scores and "
                "document pairs before treating them as duplicates."
            )

        else:

            st.info(
                "No near-duplicate pairs were found "
                "at the selected threshold."
            )


# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Text Mining & Document Intelligence System | "
    "BBC News Dataset | Corpus Analytics and Preprocessing"
)
