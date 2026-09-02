import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


# -----------------------------
# Load O*NET files
# -----------------------------

occupation_df = pd.read_csv("occupation_data.csv")
software_df = pd.read_csv("software_skills.csv")
skills_df = pd.read_csv("essential_skills.csv")
education_df = pd.read_csv("education.csv")
jobzone_df = pd.read_csv("job_zones.csv")
tasks_df = pd.read_csv("task_statements.csv")
interests_df = pd.read_csv("career_interest_types.csv")


# -----------------------------
# Create lookup dictionaries
# -----------------------------

software_data = (
    software_df
    .groupby("O*NET-SOC Code")["Workplace Example"]
    .apply(lambda x: ", ".join(x.dropna().astype(str).unique()))
    .to_dict()
)


skills_data = (
    skills_df
    .groupby("O*NET-SOC Code")["Element Name"]
    .apply(lambda x: ", ".join(x.dropna().astype(str).unique()))
    .to_dict()
)


task_data = (
    tasks_df
    .groupby("O*NET-SOC Code")["Task"]
    .apply(lambda x: " | ".join(x.dropna().astype(str).unique()))
    .to_dict()
)


education_data = (
    education_df
    .groupby("O*NET-SOC Code")["Category"]
    .apply(lambda x: ", ".join(x.dropna().astype(str).unique()))
    .to_dict()
)


jobzone_data = (
    jobzone_df
    .drop_duplicates("O*NET-SOC Code")
    .set_index("O*NET-SOC Code")["Job Zone"]
    .to_dict()
)


interest_data = (
    interests_df
    .groupby("O*NET-SOC Code")["Element Name"]
    .apply(lambda x: ", ".join(x.dropna().astype(str).unique()))
    .to_dict()
)


# -----------------------------
# Create career documents
# -----------------------------

documents = []
ids = []
metadatas = []


for i, row in occupation_df.iterrows():

    code = row["O*NET-SOC Code"]
    title = row["Title"]
    description = row["Description"]

    software = software_data.get(code, "")
    skills = skills_data.get(code, "")
    tasks = task_data.get(code, "")
    education = education_data.get(code, "")
    job_zone = jobzone_data.get(code, "")
    interests = interest_data.get(code, "")

    document = f"""
Career: {title}

Description:
{description}

Essential Skills:
{skills}

Software and Technologies:
{software}

Typical Tasks:
{tasks}

Education:
{education}

Job Zone:
{job_zone}

Career Interests:
{interests}
"""

    documents.append(document)
    ids.append(str(i))

    metadatas.append({
        "career": title,
        "soc_code": code
    })


# -----------------------------
# Load embedding model
# -----------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -----------------------------
# Create NEW ChromaDB
# -----------------------------

print("Creating O*NET ChromaDB...")

client = chromadb.PersistentClient(
    path="./onet_chroma_db"
)


# Delete old collection if this script
# has been run before

try:
    client.delete_collection("onet_careers")
except Exception:
    pass


collection = client.create_collection(
    name="onet_careers"
)


# -----------------------------
# Generate embeddings
# -----------------------------

print("Generating embeddings...")

embeddings = embedding_model.encode(
    documents,
    batch_size=8,
    show_progress_bar=True
).tolist()


# -----------------------------
# Store in ChromaDB
# -----------------------------

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=ids,
    metadatas=metadatas
)


# -----------------------------
# Finished
# -----------------------------

print()
print("O*NET career database created successfully!")
print("Number of careers:", collection.count())
print("Database location: ./onet_chroma_db")