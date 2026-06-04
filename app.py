from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

import os

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)
import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st
client = chromadb.Client()

collection = client.get_or_create_collection(
    name="careers"
)

st.title("🤖 AI Career Advisor")

st.markdown("""
Get personalized career recommendations
and ask AI-powered questions about your career path.
""")

user_input = st.text_area(
    "Tell me your interests"
)

career_names = [
    "Data Science",
    "AI Engineer",
    "Cloud Engineer",
    "Cybersecurity"
]

career_texts = [

"""
Data Science

Skills:
Python, Statistics, Data Analysis, SQL, Machine Learning

Roadmap:
Python -> Statistics -> Pandas -> Machine Learning

Salary:
6-18 LPA

Courses:
Andrew Ng Machine Learning
Resources:
Python:
- Python for Everybody

Statistics:
- Khan Academy Statistics

Machine Learning:
- Andrew Ng ML Specialization
""",

"""
AI Engineer

Skills:
Python, Machine Learning, Deep Learning, Neural Networks

Roadmap:
Python -> Machine Learning -> Deep Learning -> MLOps

Resources:
Python:
- Python for Everybody

Machine Learning:
- Andrew Ng ML Specialization

Deep Learning:
- Deep Learning Specialization

Salary:
8-20 LPA
""",

"""
Cloud Engineer

Skills:
Linux, Networking, AWS, Azure, Docker

Roadmap:
Linux -> Networking -> AWS -> DevOps

Salary:
6-18 LPA

Courses:
AWS Cloud Practitioner
Resources:
Linux:
- Linux Journey

AWS:
- AWS Cloud Practitioner

Docker:
- Docker for Beginners
""",

"""
Cybersecurity

Skills:
Networking, Cryptography, Ethical Hacking

Roadmap:
Networking -> Security -> Ethical Hacking

Salary:
5-15 LPA

Courses:
CompTIA Security+
Resources:
Networking:
- Cisco Networking Basics

Security:
- CompTIA Security+

Ethical Hacking:
- TryHackMe
- Hack The Box
"""
]
if collection.count() == 0:
    collection.add(
        documents=career_texts,
        ids=["1", "2", "3", "4"],
        metadatas=[
            {"career": "Data Science"},
            {"career": "AI Engineer"},
            {"career": "Cloud Engineer"},
            {"career": "Cybersecurity"}
        ]
    )

if st.button("Recommend"):
    
    results = collection.query(
        query_texts=[user_input],
        n_results=3
    )

    st.session_state["results"] = results
if "results" in st.session_state:

    results = st.session_state["results"]

    st.subheader("Top Career Recommendations")

   
    for i in range(len(results["metadatas"][0])):

        career = results["metadatas"][0][i]["career"]
        
        st.success(
            f"{i+1}. {career}"
        )

        with st.expander("View Details"):
            st.write(
            results["documents"][0][i]
            )

        st.divider()
    
    st.subheader("Ask About Your Career")
    career_choice = st.selectbox(
    "Choose a Career",
    [item["career"] for item in results["metadatas"][0]],
    key="career_choice"
    )

    if st.button("Select Career"):

        selected_index = [
            item["career"]
            for item in results["metadatas"][0]
        ].index(career_choice)

        st.session_state["career_doc"] = results["documents"][0][selected_index]

        st.success(f"{career_choice} selected!")
if "career_doc" in st.session_state:

    question = st.text_input(
        "Ask a question"
    )

    if question:

        career_doc = st.session_state.get(
            "career_doc",
            ""
        )

        
        

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