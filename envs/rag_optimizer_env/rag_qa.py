"""
RAG Question Answering using HotpotQA Knowledge Base + LLM
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Import the RAG retriever
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))
from rag_retrieval import get_retriever

# Load environment variables - try multiple locations
script_dir = os.path.dirname(os.path.abspath(__file__))
env_paths = [
    os.path.join(script_dir, '.env'),
    os.path.join(script_dir, 'server', '.env'),
    os.path.join(script_dir, '..', '.env'),
]

for env_path in env_paths:
    if os.path.exists(env_path):
        print(f"Loading .env from: {env_path}")
        load_dotenv(env_path)
        break

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")

print(f"Using API: {API_BASE_URL}")
print(f"Using Model: {MODEL_NAME}")
print(f"API Key loaded: {'✓' if API_KEY else '✗'}")

if not API_KEY:
    print("\n  WARNING: No API key found!")
    print("Please set OPENAI_API_KEY or HF_TOKEN in your .env file")
    print("Example: OPENAI_API_KEY=sk-...")
    API_KEY = input("\nEnter your API key (or press Enter to exit): ").strip()
    if not API_KEY:
        print("Exiting...")
        exit(1)

def answer_question(question: str, top_k: int = 3) -> dict:
    """
    Answer a question using RAG: Retrieve relevant docs + LLM generation.
    
    Args:
        question: The question to answer
        top_k: Number of documents to retrieve
        
    Returns:
        Dict with 'question', 'retrieved_docs', 'context', and 'answer'
    """
    # Initialize retriever
    retriever = get_retriever()
    retriever.initialize()
    
    # Retrieve relevant documents
    print(f"\n Retrieving top-{top_k} documents for: {question}")
    retrieved_docs = retriever.retrieve(question, top_k=top_k)
    
    # Display retrieved documents
    print("\n Retrieved Documents:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"  {i}. {doc['title']} (score: {doc['score']:.3f})")
        print(f"     {doc['text'][:100]}...")
    
    # Format context for LLM
    context = retriever.format_context(retrieved_docs)
    
    # Build prompt
    prompt = f"""Answer the question based on the provided context. If the answer cannot be found in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""
    
    # Call LLM
    print(f"\n Generating answer with {MODEL_NAME}...")
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.3
    )
    
    answer = completion.choices[0].message.content
    
    return {
        "question": question,
        "retrieved_docs": retrieved_docs,
        "context": context,
        "answer": answer
    }


def main():
    """Interactive RAG QA demo."""
    print("="*60)
    print("RAG Question Answering System")
    print("Using HotpotQA Knowledge Base")
    print("="*60)
    
    # Example questions
    example_questions = [
        "What nationality was the director of the film Titanic?",
        "Who created the character that made his debut in Action Comics #1?",
        "What is the birthplace of the actor who played in The Shawshank Redemption?"
    ]
    
    print("\nExample questions:")
    for i, q in enumerate(example_questions, 1):
        print(f"  {i}. {q}")
    
    print("\n" + "="*60)
    
    # You can either use example questions or input your own
    while True:
        print("\n" + "-"*60)
        choice = input("\nEnter question number (1-3), type your own question, or 'quit': ").strip()
        
        if choice.lower() in ['quit', 'q', 'exit']:
            break
        
        # Parse choice
        if choice.isdigit() and 1 <= int(choice) <= len(example_questions):
            question = example_questions[int(choice) - 1]
        elif choice:
            question = choice
        else:
            continue
        
        try:
            result = answer_question(question)
            
            print("\n" + "="*60)
            print(f" ANSWER: {result['answer']}")
            print("="*60)
            
        except Exception as e:
            print(f"\n Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
