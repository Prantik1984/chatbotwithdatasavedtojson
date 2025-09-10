from typing import List, Dict
from dotenv import load_dotenv
import os
import  json
import requests

history_file=""
SYSTEM_PROMPT = (
    "You are a helpful, concise coding assistant. "
    "Prefer short, correct answers with code examples when appropriate."
)

def load_history() -> List[Dict[str, str]]:
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            return json.load(f)

    return [{"role": "system", "content":SYSTEM_PROMPT}]

def approximate_token_count(text: str) -> int:
    return max(1, len(text) // 4)

def trim_history(history: List[Dict[str, str]], max_tokens: int = 3000) -> List[Dict[str, str]]:
    if not history:
        return history
    sys = history[0] if history[0].get("role") == "system" else None
    msgs = history[1:] if sys else history[:]

    contents = []
    if sys:
        for m in [sys]:
            contents.append(m.get("content", ""))

    joined = "".join(contents)
    total = approximate_token_count(joined)

    kept = []
    for m in reversed(msgs):
        t = approximate_token_count(m.get("content", ""))
        if total + t > max_tokens and kept:
            break
        kept.append(m)
        total += t
    kept.reverse()
    return ([sys] if sys else []) + kept

def save_history(history: List[Dict[str, str]]) -> None:
   with open(history_file, "w", encoding="utf-8") as f:
       json.dump(history, f, ensure_ascii=False, indent=2)

def chat_once(history: List[Dict[str, str]], user_text: str) -> str:
    history.append({"role": "user", "content":user_text})
    history=trim_history(history)

    payload = {
        "model": os.getenv("MODEL"),
        "messages": history,
        "stream": True,
        # Optional: tweak generation behavior
        "options": {
             "temperature": 0.3,
              "top_p": 0.9,
              "num_ctx": 4096,
        },
    }

    with requests.post(os.getenv("OLLAMA_URL"), json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        assistant_text = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue;
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            delta = ""
            if "message" in chunk and isinstance(chunk["message"], dict):
                delta = chunk["message"].get("content", "")
            elif "response" in chunk:
                delta = chunk.get("response", "")

            if delta:
                print(delta, end="", flush=True)
                assistant_text += delta

            if chunk.get("done"):
                break
    print()
    history.append({"role": "assistant", "content": assistant_text})
    save_history(history)

def reset_history():
    save_history([{"role": "system", "content": SYSTEM_PROMPT}])
    print("History cleared.")

def main():
    global history_file
    history_file=os.path.join(os.path.dirname(os.path.abspath(__file__)),os.getenv("HISTORY_FILE"))
    history=load_history()
    print("Simple Ollama Chatbot (context saved to chat_history.json)")
    print("Commands: /reset to clear history, /exit to quit.\n")

    while True:
        try:
            user_text=input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_text.lower() == "/exit":
            print("Bye!")
            break

        if not user_text:
            continue

        if user_text.lower() == "/reset":
            reset_history()
            history = load_history()
            continue
        chat_once(history, user_text)
if __name__ == "__main__":
    load_dotenv()
    main()