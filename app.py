# ======================================================================================
# 1. IMPORT LIBRARIES
# ======================================================================================
import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ======================================================================================
# 2. LOAD ARTIFACTS (MODEL, INDEX, DATA)
# ======================================================================================
# Use Streamlit's caching to load these heavy components only once.
@st.cache_resource
def load_artifacts():
    """
    Loads the sentence transformer model, FAISS index, and movie data.
    This function is cached to ensure it runs only once per session.
    """
    print("Loading artifacts...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index('movie_index.faiss')
    df = pd.read_csv('processed_movies.csv')
    print("Artifacts loaded successfully.")
    return model, index, df

# ======================================================================================
# 3. DEFINE CORE FUNCTIONS
# ======================================================================================
def retrieve_movies(query, model, index, df, k=5):
    """
    This is the "RETRIEVAL" part of RAG.
    It takes a user query, converts it to an embedding, and searches the FAISS
    index for the top 'k' most similar movies.
    """
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, k)
    # The indices array contains the positions of the matched movies in our dataframe.
    retrieved_docs = df.iloc[indices[0]]
    return retrieved_docs

# ======================================================================================
# 4. MAIN APPLICATION LOGIC
# ======================================================================================

# --- Load all necessary components ---
model, index, df = load_artifacts()

# --- Set up LangChain with Ollama ---
# This connects to your local Ollama model. Make sure Ollama is running!
llm = Ollama(model="llama3")

# This is the prompt template. It's a set of instructions for the AI.
# It tells the model its persona ("CineMate"), the user's question, and the
# context it should use to answer.
prompt_template = """
You are CineMate, a friendly and expert AI movie recommender.
A user has asked for a movie recommendation with the following query: "{query}"

Based on the following retrieved movies and their descriptions, provide a conversational and helpful recommendation.
You must recommend ONLY ONE movie from the list. Explain in one or two sentences WHY it is a great match for the user's request.

Retrieved movies:
{retrieved_context}

Your final recommendation:
"""

# Create the LangChain chain, which combines the prompt and the LLM.
prompt = PromptTemplate(template=prompt_template, input_variables=["query", "retrieved_context"])
chain = LLMChain(llm=llm, prompt=prompt)

# ======================================================================================
# 5. STREAMLIT USER INTERFACE
# ======================================================================================
st.set_page_config(page_title="CineMate", page_icon="🎬")
st.title("🎬 CineMate: Your AI Movie Recommender")
st.write("Powered by a local Ollama model. Tell me what you're in the mood for!")

# Get user input from a text box.
user_query = st.text_input("e.g., a mind-bending sci-fi movie with a great twist", "")

# Create a button to trigger the recommendation process.
if st.button("Get Recommendation"):
    if user_query:
        # --- Start the RAG process when the button is clicked ---
        with st.spinner("Step 1: Searching for the best movie matches..."):
            # 1. RETRIEVAL: Find relevant movies.
            retrieved_movies = retrieve_movies(user_query, model, index, df)
            # Format the retrieved data into a clean string for the LLM.
            context_for_llm = "\n".join([f"- Title: {row['title']}\n  Description: {row['cleaned_tags']}" for _, row in retrieved_movies.iterrows()])

        with st.spinner("Step 2: Asking the local AI for a personalized recommendation..."):
            # 2. GENERATION: Use the LLM to generate a conversational answer.
            # We use chain.invoke() which passes the variables to the prompt.
            response = chain.invoke({"query": user_query, "retrieved_context": context_for_llm})
        
        st.success("Here's a recommendation for you!")
        # The response from invoke is a dictionary; we extract the 'text' value.
        st.markdown(response['text'])
    else:
        st.warning("Please enter a description or movie title to get a recommendation.")

