from sentence_transformers import SentenceTransformer
import numpy as np
import pinecone

API_KEY = "api_key"
INDEX_NAME = "govtschemedata"
ENVIRONMENT = "present in host link"
# 1. Initialize Pinecone
pc = Pinecone(api_key=API_KEY)

index = pc.Index(INDEX_NAME)

# 2. Load embedding model
model = SentenceTransformer("BAAI/bge-large-en")

# 3. Prepare query
query_text = "Represent this question for retrieving relevant documents: scheduled caste in telangana"
query_embedding = model.encode(query_text, normalize_embeddings=True)  # cosine requires normalization

# 4. Query Pinecone
top_k = 5
results = index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)

# 5. Display results
for match in results['matches']:
    print(f"Score: {match['score']}")
    print(f"Document: {match['metadata'].get('text')}\n")
