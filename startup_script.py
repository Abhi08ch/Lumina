#!/usr/bin/env python3
"""
Startup script for the Document Chat application
"""
import os
import sys
import subprocess
from pathlib import Path
import time

def check_ollama():
    """Check if Ollama is installed and running"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is installed and running")
            models = result.stdout
            if "llama3" in models:
                print("✅ Llama3 model is available")
            else:
                print("⚠️  Llama3 model not found. Pulling it now...")
                subprocess.run(['ollama', 'pull', 'llama3:8b-instruct-q4_K_M'])
            return True
        else:
            print("❌ Ollama is installed but not running")
            print("💡 Try running: ollama serve")
            return False
    except FileNotFoundError:
        print("❌ Ollama is not installed")
        return False

def check_dependencies():
    """Check if required Python packages are installed"""
    required_packages = ['flask', 'sentence_transformers', 'faiss', 'fitz', 'requests', 'numpy']
    missing = []
    
    for package in required_packages:
        try:
            if package == 'fitz':
                import fitz  # PyMuPDF
            else:
                __import__(package.replace('-', '_'))
            print(f"✅ {package} is installed")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} is missing")
    
    if missing:
        print(f"\n🔧 Install missing packages with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True

def setup_directories():
    """Create necessary directories"""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    print(f"✅ Created {templates_dir} directory")
    
    # Create a simple HTML template if it doesn't exist
    html_file = templates_dir / "index.html"
    if not html_file.exists():
        html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>Document Chat</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }
        .chat-area { border: 1px solid #ccc; height: 400px; overflow-y: scroll; padding: 10px; margin: 20px 0; }
        .input-area { display: flex; gap: 10px; }
        .input-area input { flex: 1; padding: 10px; }
        .input-area button { padding: 10px 20px; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user { background: #e3f2fd; text-align: right; }
        .bot { background: #f5f5f5; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 Document Chat</h1>
        
        <div class="upload-area" id="uploadArea">
            <p>📄 Drag & drop a PDF file here or click to select</p>
            <input type="file" id="fileInput" accept=".pdf" style="display: none;">
            <button onclick="document.getElementById('fileInput').click();">Choose PDF</button>
        </div>
        
        <div id="status"></div>
        
        <div class="chat-area" id="chatArea">
            <div class="message bot">
                <strong>Bot:</strong> <span id="greeting">Loading...</span>
            </div>
        </div>
        
        <div class="input-area">
            <input type="text" id="questionInput" placeholder="Ask a question about your document..." disabled>
            <button onclick="askQuestion()" id="askBtn" disabled>Ask</button>
        </div>
    </div>

    <script>
        let fileUploaded = false;

        // Load greeting
        fetch('/greet')
            .then(r => r.json())
            .then(data => {
                document.getElementById('greeting').textContent = data.greeting;
            });

        // File upload handling
        document.getElementById('fileInput').onchange = function(e) {
            const file = e.target.files[0];
            if (file) uploadFile(file);
        };

        function uploadFile(file) {
            const formData = new FormData();
            formData.append('pdf', file);
            
            document.getElementById('status').innerHTML = '⏳ Uploading and processing...';
            
            fetch('/upload', { method: 'POST', body: formData })
                .then(r => r.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('status').innerHTML = `❌ Error: ${data.error}`;
                    } else {
                        document.getElementById('status').innerHTML = `✅ Document processed! ${data.chunk_count} chunks extracted.`;
                        fileUploaded = true;
                        document.getElementById('questionInput').disabled = false;
                        document.getElementById('askBtn').disabled = false;
                    }
                })
                .catch(err => {
                    document.getElementById('status').innerHTML = `❌ Upload failed: ${err}`;
                });
        }

        function askQuestion() {
            const question = document.getElementById('questionInput').value.trim();
            if (!question || !fileUploaded) return;
            
            // Add user message
            const chatArea = document.getElementById('chatArea');
            chatArea.innerHTML += `<div class="message user"><strong>You:</strong> ${question}</div>`;
            
            // Clear input
            document.getElementById('questionInput').value = '';
            
            // Show loading
            chatArea.innerHTML += `<div class="message bot" id="loadingMsg"><strong>Bot:</strong> ⏳ Thinking...</div>`;
            chatArea.scrollTop = chatArea.scrollHeight;
            
            fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question })
            })
            .then(r => r.json())
            .then(data => {
                document.getElementById('loadingMsg').remove();
                if (data.error) {
                    chatArea.innerHTML += `<div class="message bot"><strong>Bot:</strong> ❌ ${data.error}</div>`;
                } else {
                    chatArea.innerHTML += `<div class="message bot"><strong>Bot:</strong> ${data.answer}</div>`;
                }
                chatArea.scrollTop = chatArea.scrollHeight;
            })
            .catch(err => {
                document.getElementById('loadingMsg').remove();
                chatArea.innerHTML += `<div class="message bot"><strong>Bot:</strong> ❌ Error: ${err}</div>`;
            });
        }

        // Enter key support
        document.getElementById('questionInput').onkeypress = function(e) {
            if (e.key === 'Enter') askQuestion();
        };
    </script>
</body>
</html>'''
        html_file.write_text(html_content)
        print(f"✅ Created {html_file}")

def main():
    print("🚀 Starting Document Chat Application")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n🔧 Please install missing dependencies first!")
        sys.exit(1)
    
    # Check Ollama
    if not check_ollama():
        print("\n📋 To install Ollama:")
        print("1. Visit: https://ollama.ai")
        print("2. Download and install Ollama")
        print("3. Run: ollama serve")
        print("4. Run: ollama pull llama3:8b-instruct-q4_K_M")
        print("5. Then restart this application")
        return
    
    # Setup directories
    setup_directories()
    
    print(f"\n🌟 Starting Flask server...")
    print(f"🔗 Open your browser to: http://localhost:5000")
    print(f"\n💡 Debug endpoints:")
    print(f"   • http://localhost:5000/debug/test-ollama")
    print(f"   • http://localhost:5000/debug/chunks (after upload)")
    print(f"\n⏹️  Press Ctrl+C to stop the server")
    
    # Start the Flask server
    try:
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")
    except FileNotFoundError:
        print("\n❌ app.py not found in current directory!")
        print("💡 Make sure you're running this from the project directory")

if __name__ == "__main__":
    main()