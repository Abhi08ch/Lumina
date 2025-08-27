from flask import Flask, render_template, request, jsonify
import tempfile, os
import numpy as np
from document_utils import pdf_to_chunks
from faiss_utils import build_faiss_index, search_index
from ollama_utils import query_ollama, OLLAMA_MODEL
from sentence_transformers import SentenceTransformer
import traceback
import json
import re

app = Flask(__name__)

# Global state
chunks = None
faiss_index = None
embedding_model = None


@app.errorhandler(Exception)
def handle_exception(e):
    print("Server error:", e)
    traceback.print_exc()
    return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/greet", methods=["GET"])
def greet():
    try:
        prompt = "Say hello and ask the user to upload a PDF so you can help answer questions about it. Keep it short and friendly."
        greeting = query_ollama(prompt, model=OLLAMA_MODEL) or ""
        return jsonify({"greeting": greeting})
    except Exception:
        return jsonify({"greeting": "👋 Hi! Upload a PDF to get started."})


@app.route("/upload", methods=["POST"])
def upload_pdf():
    global chunks, faiss_index, embedding_model
    file = request.files.get("pdf")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    file.save(tf.name)
    tf.close()

    try:
        chunks = pdf_to_chunks(tf.name)
        print(f"✅ Extracted {len(chunks)} chunks from PDF")
        for i, chunk in enumerate(chunks[:3]):
            print(f"Chunk {i}: {chunk[:200]}...")
        if embedding_model is None:
            print("🔄 Loading embedding model...")
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("🔄 Creating embeddings...")
        embeds = embedding_model.encode(chunks, convert_to_numpy=True)
        faiss_index = build_faiss_index(embeds)
        print(f"✅ Successfully processed PDF with {len(chunks)} chunks")
        return jsonify({"status": "ok", "chunk_count": len(chunks)})
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tf.name)
        except Exception:
            pass


# Robust JSON-with-span extractor that locates the JSON in free text and returns parsed JSON + span
def _extract_json_span(text):
    """
    Find the first balanced JSON object (starting at a '{') in text and return (parsed_obj, start_idx, end_idx).
    Returns (None, -1, -1) if none found or parsing fails.
    """
    if not text or not isinstance(text, str):
        return None, -1, -1

    # Try to find markers first (we ask the model to put JSON between markers)
    # Common markers we may instruct: <<<JSON_START>>>, <<<JSON_END>>>, ```json ... ```
    # If markers are present, prefer them.
    start_marker = "<<<JSON_START>>>"
    end_marker = "<<<JSON_END>>>"
    if start_marker in text and end_marker in text:
        s = text.find(start_marker) + len(start_marker)
        e = text.find(end_marker, s)
        if e > s:
            candidate = text[s:e].strip()
            try:
                parsed = json.loads(candidate)
                return parsed, s, e
            except Exception:
                # fallthrough to generic search
                pass

    # Fenced code block search ```json ... ```
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if m:
        cand = m.group(1)
        try:
            parsed = json.loads(cand)
            start = m.start(1)
            end = m.end(1)
            return parsed, start, end
        except Exception:
            pass

    # Generic brace-matching approach: find first '{' and attempt to find matching '}'
    first = text.find('{')
    if first == -1:
        return None, -1, -1

    # walk through text counting braces to find matching end
    depth = 0
    end_idx = -1
    for i in range(first, len(text)):
        ch = text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    if end_idx == -1:
        return None, -1, -1

    candidate = text[first:end_idx + 1]
    try:
        parsed = json.loads(candidate)
        return parsed, first, end_idx + 1
    except Exception as e:
        # parsing failed; return None
        return None, -1, -1


def _extract_sources_from_text(text):
    """
    Simple heuristic parser to pull out 'Source N:' snippets if the model included them in free text.
    Returns list of short strings (<= 120 chars).
    """
    if not text or not isinstance(text, str):
        return []
    sources = []
    # find patterns like "Source 1: ...", "Source 2: ..." (case insensitive)
    for match in re.finditer(r"(Source\s*\d+\s*[:\-]\s*)([^\n\r]+)", text, flags=re.IGNORECASE):
        snippet = match.group(2).strip()
        # truncate to 120 chars
        snippet = snippet[:120]
        sources.append(snippet)
    return sources


def _build_soft_prompt(user_question, retrieved_chunks, conversation_history, max_chunks=6):
    """
    Compose a soft-RAG prompt:
      - system instruction allows using documents but also permits general knowledge & elaboration
      - includes concise retrieved snippets (numbered)
      - appends recent conversation turns to provide multi-turn context
      - asks the model to answer helpfully, cite docs as "Source N" when used, and include a JSON object
        after the natural language answer. The JSON must be either in a fenced ```json block or between
        the explicit markers <<<JSON_START>>> and <<<JSON_END>>> so we can reliably parse it.
    """
    system = (
        "You are Lumina, a helpful, conversational assistant specialized in helping users understand "
        "uploaded documents. Use any provided document excerpts for factual support and cite them inline "
        "as 'Source 1', 'Source 2', etc., when you rely on them. You may also use your general knowledge "
        "to elaborate, provide examples, or suggest follow-ups. If you are not sure about a fact, say so "
        "and indicate uncertainty.\n\n"
        "Answer naturally in plain language first (a few paragraphs, lists, examples). After your natural-language "
        "answer, include a JSON object that summarizes the response in this exact shape:\n"
        '{"answer": "<short summary sentence>", "sources": ["Source 1: ...", ...]}\n\n'
        "Place the JSON either inside a fenced block ```json ... ``` or between the markers <<<JSON_START>>> and <<<JSON_END>>>. "
        "This makes it easier for the system to extract the JSON programmatically. Keep the JSON truthful to the explanation.\n"
    )

    # Prepare retrieved context (trim and limit)
    ctx_lines = []
    for i, chunk in enumerate((retrieved_chunks or [])[:max_chunks]):
        src_label = f"Source {i+1}"
        snippet = chunk.strip().replace("\n", " ")
        # trim chunk to avoid huge prompts
        snippet = snippet[:1500]
        ctx_lines.append(f"{src_label} ({chunk[:60].strip() + '...' if len(chunk) > 60 else ''}):\n{snippet}")

    if ctx_lines:
        context_block = "DOCUMENT EXCERPTS (numbered):\n\n" + "\n\n---\n\n".join(ctx_lines) + "\n\n"
    else:
        context_block = "DOCUMENT EXCERPTS: None found for this query.\n\n"

    # Conversation history
    history_block = ""
    if conversation_history:
        history_lines = []
        for turn in conversation_history[-8:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            history_lines.append(f"[{role.upper()}] {content}")
        history_block = "RECENT CONVERSATION HISTORY:\n" + "\n".join(history_lines) + "\n\n"

    # Final prompt assembly
    prompt = "\n\n".join([system, context_block, history_block, f"QUESTION: {user_question}\n\nANSWER:"])
    return prompt


def _normalize_and_extract_indices(inds):
    """
    Robustly normalize the result from search_index into a flat Python list of integer indices.
    Handles:
      - None -> []
      - numpy arrays
      - lists/tuples
      - tuples like (ids, distances)
      - scalars
    """
    if inds is None:
        return []
    # If it's a tuple/list and the first element looks like ids (common for (ids, distances))
    if isinstance(inds, (tuple, list)) and len(inds) >= 1:
        first = inds[0]
        # If first element is itself an array-like (ids), prefer it
        try:
            arr = np.array(first)
            if arr.size > 0:
                return [int(x) for x in arr.reshape(-1).tolist()]
        except Exception:
            pass
    # Otherwise try to convert inds itself to array
    try:
        arr = np.array(inds)
        if arr.size == 0:
            return []
        return [int(x) for x in arr.reshape(-1).tolist()]
    except Exception:
        # As last resort, try casting to int (single scalar)
        try:
            return [int(inds)]
        except Exception:
            return []


@app.route("/ask", methods=["POST"])
def ask():
    """
    Hybrid RAG + multi-turn chat "ask" endpoint.

    Expects JSON:
      {
        "question": "User's question",
        "history": [{"role":"user"|"assistant", "content":"..."}]  // optional
      }

    Behavior:
      - Retrieve top_k chunks using FAISS (top_k=6).
      - Build a soft prompt that includes retrieved snippets and recent conversation history.
      - Allow the model to use general knowledge to elaborate while encouraging citations to the snippets.
      - Return a clean separation: 'answer' (natural-language reply), 'structured' (parsed JSON if supplied), 'sources', and 'raw'.
    """
    global chunks, faiss_index, embedding_model
    data = request.get_json(silent=True) or {}
    msg = (data.get("question") or "").strip()
    conversation_history = data.get("history", []) or []

    if not msg:
        return jsonify({"error": "Empty question"}), 400
    if not (chunks and faiss_index is not None and embedding_model is not None):
        return jsonify({"error": "No PDF uploaded yet"}), 400

    try:
        print(f"🔍 Question: {msg}")
        # create query embedding and search
        q_emb = embedding_model.encode([msg], convert_to_numpy=True)
        top_k = 6
        raw_inds = search_index(faiss_index, q_emb.astype("float32"), top_k=top_k)

        # Normalize indices safely
        inds_list = _normalize_and_extract_indices(raw_inds)
        # Filter to valid chunk indices
        retrieved = [chunks[i] for i in inds_list if isinstance(i, int) and 0 <= i < len(chunks)]
        print("📄 Retrieved chunks:")
        if retrieved:
            for i, chunk in enumerate(retrieved):
                print(f"  Chunk {i+1}: '{chunk[:120]}...'")
        else:
            print("  (no relevant chunks returned)")

        # Build a softer prompt that includes context + conversation history
        prompt = _build_soft_prompt(msg, retrieved, conversation_history, max_chunks=top_k)
        print(f"📋 Built prompt length: {len(prompt)} chars (sending to model)")

        # tuning: allow some creativity so model can elaborate
        raw = query_ollama(prompt, model=OLLAMA_MODEL, temperature=0.6, max_tokens=700, top_p=0.95, top_k=40)
        if not raw or not raw.strip():
            print("⚠️ Model returned empty response")
            return jsonify({"error": "No response from LLM (check Ollama server)."}), 502

        print(f"✅ Raw LLM output (truncated): {raw[:500]}")

        # Try to parse JSON object from the model output and also capture its span
        structured, start_idx, end_idx = _extract_json_span(raw)
        natural_answer = None

        if structured is not None:
            # Prefer the explicit natural-language portion before the JSON as the main 'answer' to render
            if start_idx > 0:
                natural_answer = raw[:start_idx].strip()
            # If the JSON itself contains an "answer" field use it as short summary fallback
            if not natural_answer:
                natural_answer = structured.get("answer", "").strip() if isinstance(structured.get("answer", ""), str) else ""
            # Normalize sources if present
            sources = structured.get("sources", [])
            if isinstance(sources, list):
                cleaned_sources = [str(s)[:120] for s in sources]
            else:
                cleaned_sources = []
            return jsonify({
                "answer": natural_answer or "",
                "structured": structured,
                "sources": cleaned_sources,
                "raw": raw
            })

        # If no JSON, attempt to heuristically extract any "Source N:" snippets
        heuristic_sources = _extract_sources_from_text(raw)
        # As fallback, if no heuristic sources, include top-k chunk starts as informal sources
        if not heuristic_sources and retrieved:
            fallback_sources = []
            for i, chunk in enumerate(retrieved[:min(5, len(retrieved))]):
                s = chunk.replace("\n", " ").strip()[:120]
                fallback_sources.append(f"Source {i+1}: {s}")
            heuristic_sources = fallback_sources

        # No structured JSON found: return raw answer and heuristic sources
        return jsonify({
            "answer": raw.strip(),
            "structured": None,
            "sources": heuristic_sources,
            "raw": raw
        })

    except Exception as e:
        print(f"❌ Error in ask endpoint: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# Debug endpoints
@app.route("/debug/chunks", methods=["GET"])
def debug_chunks():
    global chunks
    if not chunks:
        return jsonify({"error": "No PDF uploaded"})
    return jsonify({
        "total_chunks": len(chunks),
        "chunks_preview": [
            {"index": i, "content": chunk[:300] + "..." if len(chunk) > 300 else chunk, "length": len(chunk)}
            for i, chunk in enumerate(chunks[:10])
        ]
    })


@app.route("/debug/test-ollama", methods=["GET"])
def debug_test_ollama():
    test_prompt = "What is 2+2? Answer with just the number."
    response = query_ollama(test_prompt, model=OLLAMA_MODEL)
    return jsonify({
        "model": OLLAMA_MODEL,
        "prompt": test_prompt,
        "response": response,
        "working": response is not None and response.strip() != ""
    })


if __name__ == "__main__":
    app.run(debug=True)