# 🌟 Lumina — Document Chat (Local RAG)

> **A local-first, privacy-focused document chat app** using **PyMuPDF**, **SentenceTransformers**, **FAISS**, and **Ollama LLMs**.  

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Local-First](https://img.shields.io/badge/Local-First-orange)
![Ollama](https://img.shields.io/badge/LLM-Ollama-ff69b4)

---

## 🚀 Features
- 💻 **Local-first**: Your data stays on your machine.
- 🧠 **LLM-powered**: Works with **Ollama** for local inference.
- 📄 **Multi-PDF Upload**: Chat with multiple documents at once.
- 🔍 **FAISS + Embeddings**: Fast vector search for context retrieval.

---

## ⚡ Quickstart (Recommended)

### 1️⃣ Create & activate a virtual environment
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
# or if you have a bootstrap script
python setup.py
```

### 3️⃣ Install & start Ollama
- [Download Ollama](https://ollama.ai)  
- Start the server:
```bash
ollama serve
```
- Pull the model:
```bash
ollama pull llama3:8b-instruct-q4_K_M
```

### 4️⃣ Start the app
```bash
python startup_script.py
# or
python app.py
```

---

## 📡 API Endpoints
| Method | Endpoint     | Description                      |
|--------|--------------|----------------------------------|
| GET    | `/greet`     | Greeting message                 |
| POST   | `/upload`    | Upload PDF(s)                    |
| POST   | `/ask`       | Ask a question (JSON expected)   |

---

## 📝 Best Practices
- ❌ **Don’t commit**:  
  - Model weights  
  - `.venv` folder  
  - Large vector DB files  
- ✅ Use **Git LFS** or external storage for big binaries.  
- 🛡 Store secrets in `.env`, never commit them.

---

## 📂 Useful Files
- `requirements.txt` — Pin dependency versions  

---

## 🤝 Contributing
- Open an **issue** or **PR**  
- Keep changes small  
- Document any new CLI flags or env vars  

---

## 📬 Contact
**Maintainer:** [@Abhi08ch](https://github.com/Abhi08ch)  

---

### ⭐ If you like this project, give it a star on GitHub!

📸 Screenshots

<img width="1763" height="1923" alt="image" src="https://github.com/user-attachments/assets/c582e75c-1d3b-4bdc-a9af-b6d92b40466a" />

