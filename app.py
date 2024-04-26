import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

custom_css = """
<style>
.title {
    color: #ff0000; /* Light Pink */
}

.button {
    background-color: #00ff00; /* Green */
    color: #0000ff; /* Blue */
}

.text {
    color: #ffa500; /* black */
}
</style>
"""

# initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="eng_subtitles")

# distilbert-base-nli-mean-tokens model for embedding function
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="bert-base-nli-mean-tokens"
)

# get or create the collection
collection = chroma_client.get_or_create_collection(
    name="eng_subtitles",
    embedding_function=sentence_transformer_ef,
    metadata={"hnsw:space": "cosine"},
)


def main():
    st.markdown(custom_css, unsafe_allow_html=True)

    # Title with mixed color
    st.markdown(
        "<h1 class='title'>📽 Subtitle Search: The Movie Discoverer! 🍿</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p class='text'>Your favorite films with their subtitles, just a search away! 🤩</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    # getting the user input
    user_query = st.text_input("Hand over a subtitle excerpt, and the movie is yours!🔎")

    if st.button("Search"):
        if user_query:
            results = collection.query(
                query_texts=[user_query],
                n_results=10,
                include=["documents", "distances", "metadatas"],
            )
            st.markdown("---")
            # Display user input
            st.write(f"Your search query: {user_query}")

            # Display search results
            st.write("Matching Films🎬:")
            for i, metadata in enumerate(results["metadatas"][0], 1):
                metadata_value = list(metadata.values())[0]
                st.write(f"{i}. {metadata_value}")


if __name__ == "__main__":
    main()