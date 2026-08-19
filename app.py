from dotenv import load_dotenv
import os

import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st

with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# Load environment variables
load_dotenv()


# Configure Gemini
genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)


# --------------------------------------------------
# Critic Agent
# --------------------------------------------------

def critic_agent(user_query, candidates):

    candidate_text = ""

    for candidate in candidates:

        candidate_text += f"""
Career: {candidate["career"]}

Description: {candidate["description"][:500]}

"""


    prompt = f"""
You are the final career recommendation critic for a
college student career advisor.

Student's interests:
{user_query}

Candidate careers retrieved from the O*NET career database:

{candidate_text}

Select the BEST 3 careers for this student.

IMPORTANT RULES:

1. Use ONLY careers from the candidate list.
2. NEVER invent a career name.
3. Return exactly 3 career names.
4. Understand the student's interests before selecting careers.
5. Respect things the student explicitly says they do NOT want.
6. Prefer broad and recognizable career paths.
7. Prefer realistic entry-level career paths for college students.
8. Prefer careers with common industry skills and clear learning paths.
9. Avoid extremely specialized or niche careers unless the
   student's interests clearly require them.
10. Avoid research-heavy careers unless explicitly requested.
11. Do not select multiple careers that are essentially the same role.
12. If several careers belong to the same career family,
    select only the strongest one.
13. Prefer meaningfully different career directions.
14. Do not select a career merely because its title contains
    keywords from the student's query.
15. Judge the actual career description, skills and tasks.
16. Prefer career titles that a normal college student can understand.
17. If two careers are very similar, choose the broader and
    more recognizable career.

Examples of broadly recognizable career paths include:

Software Developer
Data Scientist
Data Analyst
Machine Learning Engineer
AI Engineer
Cloud Engineer
DevOps Engineer
Cybersecurity Analyst
Database Administrator
Network Engineer

These are examples only.
Use them ONLY if they actually appear in the candidate list.

Return ONLY three career names.

Put one career name on each line.

Do not add numbers.
Do not add bullet points.
Do not add explanations.
Do not use markdown.

Candidate career names:
{[candidate["career"] for candidate in candidates]}
"""


    response = model.generate_content(prompt)

    lines = response.text.strip().split("\n")

    ranked = []

    for line in lines:

        line = line.strip()

        line = line.lstrip(
            "0123456789.-) "
        )

        if line:
            ranked.append(line)


    # Keep only careers that actually exist
    valid_names = {
        candidate["career"]
        for candidate in candidates
    }


    final_ranked = []

    for career in ranked:

        if career in valid_names:

            if career not in final_ranked:
                final_ranked.append(career)


    return final_ranked[:3]


# --------------------------------------------------
# Load Embedding Model
# --------------------------------------------------


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()



# --------------------------------------------------
# Connect to NEW O*NET ChromaDB
# --------------------------------------------------

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(
        path="./onet_chroma_db"
    )
    return client.get_collection(
        name="onet_careers"
    )


collection = load_collection()


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🤖 AI Career Advisor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Discover career paths tailored to your interests, skills and preferences.</div>',
    unsafe_allow_html=True
)


user_input = st.text_area(
    "Tell me your interests"
)


# --------------------------------------------------
# Recommend Careers
# --------------------------------------------------

if st.button("Recommend"):

    if user_input.strip():

        # Convert student query into embedding
        query_embedding = embedding_model.encode(
            user_input
        ).tolist()


        # Retrieve a larger candidate pool
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=30
        )


        # Build candidate list
        candidate_careers = []

        for i in range(
            len(results["metadatas"][0])
        ):

            metadata = results["metadatas"][0][i]

            document = results["documents"][0][i]


            candidate_careers.append({
                "career": metadata["career"],
                "soc_code": metadata["soc_code"],
                "description": document
            })


        # Remove duplicate career names
        unique_candidates = []

        seen = set()

        for candidate in candidate_careers:

            career = candidate["career"]

            if career not in seen:

                seen.add(career)

                unique_candidates.append(
                    candidate
                )


        # Send candidates to critic

        with st.spinner("🤖 Analyzing your interests..."):
            ranked_careers = critic_agent(
                user_input,
                unique_candidates
            )

        st.session_state["results"] = results

        st.session_state["ranked_careers"] = ranked_careers

    else:

        st.warning(
            "Please enter your interests first."
        )


# --------------------------------------------------
# Display Recommendations
# --------------------------------------------------

if "results" in st.session_state:

    results = st.session_state["results"]

    ranked_careers = (
        st.session_state["ranked_careers"]
    )


    # Map career name to document
    career_documents = {}

    for i in range(
        len(results["metadatas"][0])
    ):

        career = (
            results["metadatas"][0][i]["career"]
        )

        document = (
            results["documents"][0][i]
        )

        career_documents[career] = document


    st.markdown(
        '<div class="section-title">🏆 Top Career Recommendations</div>',
        unsafe_allow_html=True
    )


    # Display top 3
    displayed = 0

    for career in ranked_careers:

        career = career.strip()

        if career in career_documents:

            displayed += 1

            description = career_documents[career]

            preview = description[:220] + "..."

            st.markdown(
                f"""
                <div class="career-card">
                    <div class="career-number">🏆 #{displayed}</div>
                    <div class="career-title">{career}</div>
                    <div class="career-preview">{preview}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander("📋 View Full Career Details"):

                st.markdown("### 💼 Career Overview")
                st.write(description)

                st.markdown("### 🎯 Why This Career?")
                st.info(
                    "This career matches the interests and skills identified "
                    "from the O*NET career information."
                )

                st.markdown("### 📚 O*NET Information")
                st.caption(
                    "The details above are retrieved from the O*NET career database."
                )

            if displayed == 3:
                break


    # --------------------------------------------------
    # Ask About Career
    # --------------------------------------------------

    st.subheader(
        "Ask About Your Career"
    )


    available_careers = [
        career
        for career in ranked_careers
        if career.strip()
        in career_documents
    ][:3]


    if available_careers:

        career_choice = st.selectbox(
            "Choose a Career",
            available_careers,
            key="career_choice"
        )


        if st.button(
            "Select Career"
        ):

            selected_career = (
                career_choice.strip()
            )


            st.session_state[
                "career_doc"
            ] = career_documents[
                selected_career
            ]


            st.session_state[
                "selected_career"
            ] = selected_career


            st.success(
                f"{selected_career} selected!"
            )


# --------------------------------------------------
# Ask Questions About Selected Career
# --------------------------------------------------

if "career_doc" in st.session_state:

    with st.expander("🔍 Debug: Retrieved O*NET Data"):

        st.write(
            st.session_state["career_doc"]
        )

    question = st.text_input(
        "Ask a question"
    )

    if question:

        career_doc = st.session_state[
            "career_doc"
        ]

        prompt = f"""
You are an expert career advisor.

You are answering questions using the
following O*NET career information.

Career Information:
{career_doc}

Student Question:
{question}

Instructions:

- Answer primarily using the Career Information.
- Do not invent career-specific facts.
- If the information is insufficient,
  clearly say so.
- General career advice is allowed when
  clearly identified as general guidance.
- Give practical advice for a college student.
- Explain things in simple language.
- Keep the answer under 150 words.

Answer:
"""

        response = model.generate_content(
            prompt
        )

        st.write(
            response.text
        )