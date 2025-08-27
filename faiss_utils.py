import faiss
import numpy as np

def build_faiss_index(embeddings):
    """
    Build FAISS index from embeddings using cosine similarity
    
    Args:
        embeddings: numpy array of shape (n_docs, embedding_dim)
    
    Returns:
        FAISS index
    """
    embeddings = embeddings.astype('float32')
    dimension = embeddings.shape[1]
    
    print(f"🔄 Building FAISS index...")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Dimension: {dimension}")
    
    # Use IndexFlatIP for cosine similarity (Inner Product after normalization)
    index = faiss.IndexFlatIP(dimension)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Add embeddings to index
    index.add(embeddings)
    
    print(f"✅ Built FAISS index with {index.ntotal} vectors")
    return index

def search_index(index, query_embedding, top_k=3):
    """
    Search index and return top-k most similar document indices
    
    Args:
        index: FAISS index
        query_embedding: numpy array of shape (1, embedding_dim)
        top_k: number of results to return
    
    Returns:
        List of document indices
    """
    query_embedding = query_embedding.astype('float32')
    
    # Normalize query for cosine similarity
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = index.search(query_embedding, top_k)
    
    print(f"🔍 Search results:")
    for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
        print(f"  {i+1}. Document {idx} (similarity: {score:.3f})")
    
    return indices[0]  # Return first (and only) query's results

def search_index_with_scores(index, query_embedding, top_k=5):
    """
    Search index and return both indices and similarity scores
    
    Args:
        index: FAISS index
        query_embedding: numpy array of shape (1, embedding_dim)
        top_k: number of results to return
    
    Returns:
        Tuple of (indices, scores)
    """
    query_embedding = query_embedding.astype('float32')
    
    # Normalize query for cosine similarity  
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = index.search(query_embedding, top_k)
    
    return indices[0], scores[0]