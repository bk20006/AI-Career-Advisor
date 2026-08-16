from dotenv import load_dotenv
import os

import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st


# Load environment variables
load_dotenv()


# Configure Gemini
genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)
def critic_agent(user_query, careers):

    prompt = f"""
You are a career recommendation critic.

Student's interests:
{user_query}

Candidate careers:
{careers}

Rank the candidate careers from most suitable to least suitable.

Consider:
- Skills mentioned by the student
- Career responsibilities
- Student's explicit preferences
- Things the student explicitly does NOT want
- Overall career fit

Return ONLY the career names in ranked order.
Do not add explanations.

Example:
Cloud Engineer
Network Engineer
MLOps Engineer
"""

    response = model.generate_content(prompt)

    lines = response.text.strip().split("\n")

    ranked = []

    for line in lines:
        line = line.strip()
        line = line.lstrip("0123456789.-) ")

        if line:
            ranked.append(line)

    return ranked


# Load embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# Connect to persistent ChromaDB
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="careers"
)


# Streamlit UI
st.title("🤖 AI Career Advisor")

st.markdown("""
Get personalized career recommendations
and ask AI-powered questions about your career path.
""")


user_input = st.text_area(
    "Tell me your interests"
)


# Recommend careers
if st.button("Recommend"):

    if user_input.strip():

        # Convert user query into an embedding
        query_embedding = embedding_model.encode(
            user_input
        ).tolist()

        # Retrieve top 5 candidates from ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        candidate_careers = []

        for i in range(len(results["metadatas"][0])):
            candidate_careers.append(
                results["documents"][0][i]
            )

        # Critic agent reranks the candidates
        ranked_careers = critic_agent(
            user_input,
            candidate_careers
        )

        st.session_state["results"] = results
        st.session_state["ranked_careers"] = ranked_careers

    else:
        st.warning(
            "Please enter your interests first."
        )

# Display recommendations
if "results" in st.session_state:

    results = st.session_state["results"]
    ranked_careers = st.session_state["ranked_careers"]

    # Create a mapping between career name and its document
    career_documents = {}

    for i in range(len(results["metadatas"][0])):
        career = results["metadatas"][0][i]["career"]
        document = results["documents"][0][i]

        career_documents[career] = document

    st.subheader("Top Career Recommendations")

    # Display critic-ranked careers
    for i, career in enumerate(ranked_careers[:3]):

        # Find the matching career document
        career = career.strip()

        if career in career_documents:

            st.success(
                f"{i + 1}. {career}"
            )

            with st.expander("View Details"):
                st.write(
                    career_documents[career]
                )

            st.divider()


    st.subheader("Ask About Your Career")

    # Only show careers that were actually ranked
    available_careers = [
        career
        for career in ranked_careers[:3]
        if career.strip() in career_documents
    ]

    career_choice = st.selectbox(
        "Choose a Career",
        available_careers,
        key="career_choice"
    )


    if st.button("Select Career"):

        selected_career = career_choice.strip()

        st.session_state["career_doc"] = (
            career_documents[selected_career]
        )

        st.session_state["selected_career"] = (
            selected_career
        )

        st.success(
            f"{selected_career} selected!"
        )


# Ask questions about selected career
if "career_doc" in st.session_state:

    question = st.text_input(
        "Ask a question"
    )

    if question:

        career_doc = st.session_state[
            "career_doc"
        ]

        prompt = f"""
You are an expert career advisor.

Career Information:
{career_doc}

Student Question:
{question}

Instructions:
- Give practical advice.
- Explain the reason.
- Suggest resources if available.
- Keep answers under 150 words.
- Use simple language.

Answer:
"""

        response = model.generate_content(
            prompt
        )

        st.write(
            response.text
        )