# Helpers to build chat-style LLM messages for a RAG + multi-turn experience.
# Usage: messages = build_chat_messages(user_query, retrieved_chunks, conversation_history)
from typing import List, Dict

def build_chat_messages(user_query: str,
                        retrieved_chunks: List[Dict],
                        conversation_history: List[Dict],
                        max_history_turns: int = 6,
                        max_chunks: int = 6) -> List[Dict]:
    """
    Build a chat messages array for a chat-style LLM API.
    - retrieved_chunks: list of dicts like {"source": "file.pdf", "text": "..." , "score": 0.92}
    - conversation_history: list of {"role":"user"|"assistant","content": "..."}
    Returns: messages suitable for chat APIs (OpenAI/llama-like)
    """
    system = {
        "role": "system",
        "content": (
            "You are Lumina, a helpful expert assistant. Use any provided document excerpts for factual details and "
            "cite sources (e.g. 'Source 1') when you rely on them. You may also use your general knowledge to elaborate, "
            "explain, summarize, or provide examples. If you are not sure, say so and offer clarifying questions."
        )
    }

    # Prepare context block of retrieved chunks (concise, numbered)
    context_msgs = []
    if retrieved_chunks:
        # Limit chunks and trim text to avoid token explosion
        limited = retrieved_chunks[:max_chunks]
        context_lines = []
        for i, c in enumerate(limited):
            src = c.get("source", f"source_{i+1}")
            text = c.get("text", "")
            # Trim the snippet to avoid huge payloads (adjust length as needed)
            snippet = text[:2000].strip()
            context_lines.append(f"Source {i+1} ({src}):\n{snippet}")
        context_text = "\n\n".join(context_lines)
        context_msgs.append({
            "role": "system",
            "content": "Relevant document excerpts (only as reference):\n\n" + context_text
        })
    else:
        # No docs found - inform model but allow general knowledge
        context_msgs.append({
            "role": "system",
            "content": "No relevant documents were found for this query. Answer using your general knowledge and indicate uncertainty where appropriate."
        })

    # Re-add the recent conversation turns (keep last N)
    history_msgs = []
    for turn in (conversation_history or [])[-max_history_turns:]:
        # turn is expected to be {"role": "user"|"assistant", "content": "..."}
        history_msgs.append({"role": turn["role"], "content": turn["content"]})

    # Final user message
    user_msg = {"role": "user", "content": user_query}

    # Compose final messages list
    messages = [system] + context_msgs + history_msgs + [user_msg]
    return messages