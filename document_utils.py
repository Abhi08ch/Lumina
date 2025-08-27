import fitz  # PyMuPDF

def pdf_to_chunks(file_path, chunk_size=500, overlap=100):
    """
    Extract text from PDF and split into overlapping chunks.
    
    Args:
        file_path: Path to the PDF file
        chunk_size: Number of words per chunk
        overlap: Number of words to overlap between chunks
    
    Returns:
        List of text chunks
    """
    try:
        doc = fitz.open(file_path)
        print(f"📄 Processing PDF with {len(doc)} pages")
        
        # Extract text from all pages
        full_text = []
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            if page_text.strip():  # Only add non-empty pages
                full_text.append(page_text)
                print(f"  Page {page_num + 1}: {len(page_text)} characters")
        
        doc.close()
        
        if not full_text:
            raise ValueError("No text found in PDF")
        
        # Join all text and split into words
        combined_text = " ".join(full_text)
        words = combined_text.split()
        
        print(f"📝 Total words extracted: {len(words)}")
        
        # Create overlapping chunks
        chunks = []
        i = 0
        step = chunk_size - overlap if chunk_size > overlap else chunk_size
        
        while i < len(words):
            # Get chunk of words
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Only add non-empty chunks
            if chunk_text.strip():
                chunks.append(chunk_text)
            
            i += step
            
            # Prevent infinite loop
            if step <= 0:
                break
        
        print(f"✅ Created {len(chunks)} chunks")
        
        # Show sample chunks for debugging
        for i, chunk in enumerate(chunks[:2]):
            print(f"  Sample chunk {i+1}: '{chunk[:100]}...'")
        
        return chunks
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        raise

def clean_text(text):
    """Clean extracted text by removing extra whitespace and fixing common issues"""
    import re
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common PDF artifacts
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text