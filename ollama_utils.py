import subprocess
import sys
import json
import requests

# Recommended quantized model for speed + quality
OLLAMA_MODEL = "llama3:8b-instruct-q4_K_M"

def query_ollama_api(prompt, model=OLLAMA_MODEL, temperature=0.1, max_tokens=512, top_p=0.9, top_k=40):
    """Use Ollama API instead of subprocess for better control"""
    try:
        url = "http://localhost:11434/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_predict": max_tokens
            }
        }
        
        response = requests.post(url, json=data, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        # Ollama's API shape may differ; try a few common keys
        if isinstance(result, dict):
            # If 'response' exists, return that; else join any outputs
            return result.get("response") or result.get("text") or result.get("output") or ""
        return str(result)
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Ollama API. Is Ollama running?")
        return None
    except Exception as e:
        print(f"❌ Ollama API error: {e}")
        return None

def query_ollama_subprocess(prompt, model=OLLAMA_MODEL, temperature=None):
    """Fallback subprocess method (temperature ignored for subprocess fallback)"""
    try:
        cmd = ["ollama", "run", model]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        
        out, err = proc.communicate(prompt, timeout=120)
        
        if err and err.strip():
            print("Ollama stderr:", err, file=sys.stderr)
        
        if proc.returncode != 0:
            print(f"Ollama returned exit code: {proc.returncode}")
            return None
            
        return (out or "").strip()
        
    except subprocess.TimeoutExpired:
        proc.kill()
        print("❌ Ollama subprocess timed out")
        return None
    except Exception as e:
        print(f"❌ Ollama subprocess error: {e}")
        return None

def query_ollama(prompt, model=OLLAMA_MODEL, temperature=0.1, max_tokens=512, top_p=0.9, top_k=40):
    """
    Main query function - tries API first, falls back to subprocess.
    Accepts tuning params and forwards them to the API call.
    """
    # Try API method first (more reliable)
    result = query_ollama_api(prompt, model=model, temperature=temperature, max_tokens=max_tokens, top_p=top_p, top_k=top_k)
    if result:
        # Ollama API sometimes wraps result as dict or list; ensure string
        if isinstance(result, (dict, list)):
            try:
                return json.dumps(result)
            except Exception:
                return str(result)
        return str(result).strip()
    
    # Fallback to subprocess (no fine-grained options supported there)
    print("🔄 API failed, trying subprocess...")
    return query_ollama_subprocess(prompt, model=model, temperature=temperature)

def test_ollama_connection(model=OLLAMA_MODEL):
    """Test if Ollama is running and model is available"""
    try:
        test_response = query_ollama("Say 'OK' if you can read this.", model=model)
        return test_response is not None and test_response.strip() != ""
    except Exception as e:
        print(f"❌ Ollama connection test failed: {e}")
        return False