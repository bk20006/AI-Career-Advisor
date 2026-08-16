import json
import chromadb
from sentence_transformers import SentenceTransformer

# Load career data
with open("career_data.json", "r", encoding="utf-8") as file:
    careers = json.load(file)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get collection
collection = client.get_or_create_collection(
    name="careers"
)

# Prepare data
documents = []
ids = []
metadatas = []

for i, career in enumerate(careers):

    document = f"""
Career: {career["career"]}

Description:
{career["description"]}

Skills:
{", ".join(career["skills"])}

Roadmap:
{" -> ".join(career["roadmap"])}

Tools:
{", ".join(career["tools"])}

Projects:
{", ".join(career["projects"])}

Certifications:
{", ".join(career["certifications"])}
"""

    documents.append(document)
    ids.append(str(i))
    metadatas.append({
        "career": career["career"]
    })

# Generate embeddings
embeddings = embedding_model.encode(documents).tolist()

# Store documents and embeddings in ChromaDB
collection.upsert(
    documents=documents,
    embeddings=embeddings,
    ids=ids,
    metadatas=metadatas
)

print("Career database created successfully!")
print("Number of careers:", collection.count())