import os
import json
from openai import OpenAI

from client import RagOptimizerEnvClient
from models import RagOptimizerAction

# Load environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("OPENAI_API_KEY", ""))

MAX_STEPS = 15

SYSTEM_PROMPT = """You are an automated Data Engineer managing an AI Knowledge Base.
Your goal is to optimize the messy chunks of text in the database so that a TF-IDF Search Algorithm can find answers easily.
You must resolve contradictions, categorize documents, and splinter monolithic text blobs into smaller chunks.

You have the following actions:
- {"action_type": "read_document", "doc_id": "..."}
- {"action_type": "update_document", "doc_id": "...", "text": "..."}
- {"action_type": "delete_document", "doc_id": "..."}
- {"action_type": "add_metadata", "doc_id": "...", "metadata_key": "...", "metadata_value": "..."}
- {"action_type": "submit"}

You must return ONLY a raw JSON object detailing the action you want to take!"""

def main():
    # Setup OpenAI Client
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    # Connect to the local OpenEnv server using .sync() since EnvClient is async by default
    print("Connecting to backend OpenEnv Server at http://localhost:8000...")
    with RagOptimizerEnvClient(base_url="http://localhost:8000").sync() as env:
        
        result = env.reset()
        observation = result.observation
        
        print("\n--- ENVIRONMENT RESET ---")
        print(f"Server Message: {observation.message}")
        print(f"Initial KB Stats: {len(observation.current_docs)} files loaded.\n")
        
        # Conversation history so the LLM remembers previous actions
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add initial observation
        init_obs = {
            "server_feedback": observation.message,
            "current_knowledge_base": observation.current_docs
        }
        history.append({"role": "user", "content": json.dumps(init_obs, indent=2)})
        
        for step in range(1, MAX_STEPS + 1):
            
            # Build messages from full history
            messages = list(history)

            try:
                # Call OpenAI SDK for inference exactly like the Hackathon example
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_tokens=1000
                )
                response_text = completion.choices[0].message.content or ""
                action_data = json.loads(response_text)
                # Normalize fields if model returns lists instead of strings
                for field in ("doc_id", "text", "metadata_key", "metadata_value"):
                    val = action_data.get(field)
                    if isinstance(val, list):
                        # Join if it's a list of strings, take first if list of dicts
                        if val and isinstance(val[0], str):
                            action_data[field] = " ".join(val)
                        elif val and isinstance(val[0], dict):
                            action_data[field] = json.dumps(val[0])
                        else:
                            action_data[field] = str(val[0]) if val else ""
                action = RagOptimizerAction(**action_data)
                
            except Exception as exc: 
                print(f"Model request failed ({exc}). Using fallback action.")
                # Fallback directly to end episode so we don't crash the simulation
                action = RagOptimizerAction(action_type="submit")

            print(f"Step {step}: Agent called -> {action.action_type} on doc_id: {action.doc_id}")

            # Send action to Environment
            result = env.step(action)
            observation = result.observation
            reward = result.reward

            # Append this turn to history so the LLM remembers what it did
            history.append({"role": "assistant", "content": json.dumps(action.model_dump(), default=str)})
            next_obs = {
                "server_feedback": observation.message,
                "current_knowledge_base": observation.current_docs
            }
            history.append({"role": "user", "content": json.dumps(next_obs, indent=2)})

            print(f"  Reward Status: {reward:+.2f} | Done: {result.done} | Server: {observation.message}")
            print("\n")
            
            if result.done:
                print("\n=== EPISODE COMPLETE ===")
                print(f"Final Hackathon Score: {reward * 100:.1f}%")
                break

        else:
            print(f"Reached max steps ({MAX_STEPS}). Exiting.")
            print(f"Final Score: {result.reward * 100:.1f}%")

if __name__ == "__main__":
    main()
