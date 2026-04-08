import os
import sys
import json
from openai import OpenAI

# Add the envs module to path so we can import client and models
sys.path.append(os.path.join(os.path.dirname(__file__), "envs", "rag_optimizer_env"))
from client import RagOptimizerEnvClient
from models import RagOptimizerAction

# Load environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("OPENAI_API_KEY", ""))

MAX_STEPS = 30

SYSTEM_PROMPT = """You are an automated Data Engineer managing an AI Knowledge Base.
Your goal is to optimize the messy chunks of text in the database so that a TF-IDF Search Algorithm can find answers easily.
You must resolve contradictions, categorize documents, and delete unnecessary documents.

After each action you will receive a "current_reward" score (0.0 to 1.0) indicating how well the KB currently performs. Use this to guide your strategy.

You have the following actions:
- {"action_type": "read_document", "doc_id": "..."}
- {"action_type": "update_document", "doc_id": "...", "text": "..."}
- {"action_type": "delete_document", "doc_id": "..."}
- {"action_type": "add_metadata", "doc_id": "...", "metadata_key": "...", "metadata_value": "..."}
- {"action_type": "submit"}

You must return ONLY a raw JSON object detailing the action you want to take!"""

def format_action_str(action: RagOptimizerAction) -> str:
    if action.action_type == "read_document":
        return f"read('{action.doc_id}')"
    elif action.action_type == "update_document":
        return f"update('{action.doc_id}')"
    elif action.action_type == "delete_document":
        return f"delete('{action.doc_id}')"
    elif action.action_type == "add_metadata":
        return f"add_metadata('{action.doc_id}','{action.metadata_key}')"
    elif action.action_type == "submit":
        return "submit()"
    return f"{action.action_type}()"

def main():
    # Setup OpenAI Client
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    # Track metrics for the final output
    step_rewards = []
    success = False
    error_msg = "null"
    score = 0.0
    
    print(f"[START] task=rag_optimizer_env env=OpenEnv model={MODEL_NAME}")
    
    # We suppress any other custom prints to respect the STDOUT format strictly
    import contextlib
    import io
    
    with RagOptimizerEnvClient(base_url="http://localhost:8000").sync() as env:
        # Suppress prints from client or env reset
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                result = env.reset()
                observation = result.observation
            except Exception as e:
                error_msg = str(e).replace('\n', ' ')
                print(f"[END] success=false steps=0 score=0.00 rewards=")
                return

        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        init_obs = {
            "server_feedback": observation.message,
            "current_reward": observation.reward,
            "current_knowledge_base": observation.current_docs
        }
        history.append({"role": "user", "content": json.dumps(init_obs, indent=2)})
        
        step = 0
        for i in range(1, MAX_STEPS + 1):
            step = i
            messages = list(history)
            
            action_str = "unknown"
            error_msg = "null"

            try:
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
                        if val and isinstance(val[0], str):
                            action_data[field] = " ".join(val)
                        elif val and isinstance(val[0], dict):
                            action_data[field] = json.dumps(val[0])
                        else:
                            action_data[field] = str(val[0]) if val else ""
                
                action = RagOptimizerAction(**action_data)
                action_str = format_action_str(action)
                
            except Exception as exc: 
                error_msg = str(exc).replace('\n', ' ')
                action = RagOptimizerAction(action_type="submit")
                action_str = format_action_str(action)

            # Suppress normal prints during step
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    result = env.step(action)
                    observation = result.observation
                    reward = result.reward
                except Exception as e:
                    error_msg = str(e).replace('\n', ' ')
                    reward = 0.0
                    result = type('obj', (object,), {'done': True})()
                    observation = type('obj', (object,), {'message': 'error', 'current_docs': {}})()

            step_rewards.append(reward)
            done = "true" if result.done else "false"
            
            print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done} error={error_msg}")

            if result.done:
                success = True if reward > 0.5 else False  # Or however you define success
                score = float(reward)
                break

            history.append({"role": "assistant", "content": json.dumps(action.model_dump(), default=str)})
            next_obs = {
                "server_feedback": observation.message,
                "current_reward": observation.reward,
                "current_knowledge_base": observation.current_docs
            }
            history.append({"role": "user", "content": json.dumps(next_obs, indent=2)})

        else:
            # Reached max steps
            success = False
            score = float(result.reward)

    rewards_str = ",".join([f"{r:.2f}" for r in step_rewards])
    done_str = "true" if success else "false"
    print(f"[END] success={done_str} steps={step} score={score:.2f} rewards={rewards_str}")

if __name__ == "__main__":
    main()
