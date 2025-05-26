import json
import uuid
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------- Configuration ---------------- #
INPUT_FILE = "govtschmes.txt"        # 🔁 Replace with your file path
OUTPUT_FILE = "embeddings_output.json"   # 🔁 Output JSON file
SOURCE_NAME = "document-1"               # 🔁 Optional tag for source document

CHUNK_SIZE = 3000
CHUNK_OVERLAP = 1000
MODEL_NAME = "BAAI/bge-large-en-v1.5"    # or "BAAI/bge-base-en-v1.5"
# ------------------------------------------------ #

# Load model
print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, device="cuda")

# Read input text
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    full_text = f.read()

# Chunking function
def chunk_text(text, chunk_size=3000, overlap=1000):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append({
            "text": chunk,
            "character_start": start,
            "character_end": end
        })
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks

# Chunk text
print("Chunking text...")
chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)

# Add prefix for BGE
texts = ["passage: " + chunk["text"] for chunk in chunks]

# Encode
print(f"Encoding {len(texts)} chunks...")
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

# Prepare output
print("Generating JSON with embeddings and metadata...")
output_data = []
for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
    output_data.append({
        "id": str(uuid.uuid4()),
        "values": emb.tolist(),
        "metadata": {
            "source": SOURCE_NAME,
            "text": chunk["text"],
            "chunk_index": idx,
            "character_start": chunk["character_start"],
            "character_end": chunk["character_end"]
        }
    })

# Save to JSON
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

print(f"✅ Done! {len(output_data)} embeddings saved to {OUTPUT_FILE}")
