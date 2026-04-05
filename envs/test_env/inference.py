import os
import json
from openai import OpenAI
from client import TestEnv
from models import RagAction

# Configuration requirements per problem statement
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("OPENAI_API_KEY", ""))

SYSTEM_PROMPT = """You are a Data Engineer tasked with optimizing a messy Knowledge Base for RAG.
You have the following actions available:
- read_document (give doc_id)
- update_document (give doc_id and text, this creates or replaces a chunk)
- delete_document (give doc_id)
- add_metadata (give doc_id, metadata_key, metadata_value)
- submit (end episode when you are done)

Analyze the current_docs. You should update policies if they conflict. If you see messy tickets, try to add metadata or split them into cleanly formatted text blocks. You must respond in valid JSON matching the RagAction schema.
Example JSON:
{"action_type": "read_document", "doc_id": "doc_pricing_2021"}
{"action_type": "delete_document", "doc_id": "doc_pricing_2021"}
{"action_type": "update_document", "doc_id": "clean_ticket_1", "text": "Frontend team fixed the CSS button bug."}
"""

def get_llm_action(client: OpenAI, history: list) -> RagAction:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=history,
            response_format={"type": "json_object"}
        )
        action_dict = json.loads(response.choices[0].message.content)
        return RagAction(**action_dict)
    except Exception as e:
        print(f"LLM Error: {e}")
        # Default fallback to avoid crash
        return RagAction(action_type="submit")

def run():
    openai_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    # Run against local server (ensure you trigger in a separate terminal: uv run --project . server)
    with TestEnv(base_url="http://localhost:8000").sync() as env:
        print("Starting RagOptimizerEnv...")
        result = env.reset()
        obs = result.observation
        
        history = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        
        done = False
        step = 0
        while not done and step < 10:
            print(f"-- Step {step} --")
            env_state = f"Message: {obs.message}\nCurrent Docs: {json.dumps(obs.current_docs)}"
            print(env_state)
            
            history.append({"role": "user", "content": env_state})
            action = get_llm_action(openai_client, history)
            
            print(f"Agent took action: {action.action_type} on {action.doc_id}")
            
            # Record action in history
            history.append({"role": "assistant", "content": action.model_dump_json()})
            
            result = env.step(action)
            obs = result.observation
            done = result.done
            step += 1
            
        print(f"\nEpisode complete. Final Reward: {result.reward}")

if __name__ == "__main__":
    run()
