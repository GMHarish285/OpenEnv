from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
import json

# ── 1. Load HotpotQA ──────────────────────────────────────────
print("Loading HotpotQA...")
dataset = load_dataset("hotpot_qa", "distractor", split="train[:2000]")

# ── 2. Extract all unique Wikipedia paragraphs ────────────────
print("Extracting documents...")
documents = {}  # title → text  (dict to avoid duplicates)

for sample in dataset:
    titles = sample["context"]["title"]
    sentences = sample["context"]["sentences"]
    
    for title, sent_list in zip(titles, sentences):
        if title not in documents:
            # Join sentences into one paragraph
            documents[title] = " ".join(sent_list)

print(f"Unique documents extracted: {len(documents)}")

# ── 3. Load embedding model ───────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, CPU friendly

# ── 4. Initialize ChromaDB ────────────────────────────────────
client = chromadb.PersistentClient(path="./knowledge_base")  # saves to disk
collection = client.get_or_create_collection(
    name="hotpotqa_kb",
    metadata={"hnsw:space": "cosine"}  # cosine similarity for retrieval
)

# ── 5. Add documents in batches ───────────────────────────────
print("Building KB — adding documents...")

titles_list = list(documents.keys())
texts_list = list(documents.values())

BATCH_SIZE = 100

for i in range(0, len(titles_list), BATCH_SIZE):
    batch_titles = titles_list[i:i+BATCH_SIZE]
    batch_texts  = texts_list[i:i+BATCH_SIZE]
    
    # Generate embeddings
    embeddings = embedder.encode(batch_texts, show_progress_bar=False).tolist()
    
    # Clean IDs (ChromaDB doesn't like special characters)
    ids = [f"doc_{i+j}" for j in range(len(batch_titles))]
    
    # Store metadata (title so we can trace back)
    metadatas = [{"title": t} for t in batch_titles]
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=batch_texts,
        metadatas=metadatas
    )
    
    print(f"  Added {min(i+BATCH_SIZE, len(titles_list))}/{len(titles_list)} docs")

print(f"\n KB built! Total docs in KB: {collection.count()}")


# ── 6. Test retrieval ─────────────────────────────────────────
def retrieve(query, top_k=3):
    """Retrieve top-k documents for a given query."""
    # Embed the query
    query_embedding = embedder.encode(query).tolist()
    
    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        retrieved.append({
            "title": meta["title"],
            "text": doc,
            "score": 1 - dist  # convert distance to similarity
        })
    
    return retrieved


def evaluate_retrieval(n_samples=100):
    """Evaluate retrieval accuracy against ground truth supporting facts."""
    correct_retrievals = 0
    
    for sample in dataset.select(range(n_samples)):
        question = sample["question"]
        needed_titles = set(sample["supporting_facts"]["title"])
        
        # Retrieve top 3 docs
        retrieved = retrieve(question, top_k=3)
        retrieved_titles = set([d["title"] for d in retrieved])
        
        # Check overlap with ground truth
        overlap = needed_titles & retrieved_titles
        if len(overlap) == len(needed_titles):
            correct_retrievals += 1
    
    accuracy = correct_retrievals / n_samples
    print(f"Retrieval accuracy: {accuracy:.2%}")
    print(f"(found ALL needed docs for {correct_retrievals}/{n_samples} questions)")
    return accuracy


# Test it
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing retrieval...")
    print("="*60)
    
    question = "What nationality was the director of the film Titanic?"
    docs = retrieve(question, top_k=3)

    for doc in docs:
        print(f"\n {doc['title']} (score: {doc['score']:.3f})")
        print(f"   {doc['text'][:150]}...")
    
    # Run evaluation
    print("\n" + "="*60)
    print("Evaluating retrieval accuracy...")
    print("="*60)
    evaluate_retrieval()
