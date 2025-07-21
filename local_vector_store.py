import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

class LocalVectorStore:
    def __init__(self, dimension=768, store_path="vector_store"):
        self.dimension = dimension
        self.store_path = store_path
        #self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        
        # Create storage directory
        os.makedirs(store_path, exist_ok=True)
        
        # Initialize or load existing index
        self.index_file = os.path.join(store_path, "docs.index")
        self.metadata_file = os.path.join(store_path, "metadata.pkl")
        
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            self.load_index()
        else:
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            self.texts = []
            self.metadata = []
    
    def get_embedding(self, text):
        """Generate embedding using HuggingFace model"""
        embedding = self.embeddings_model.encode(text)
        return embedding
    
    def add_documents(self, texts, metadatas, embeddings=None):
        """Add documents to the vector store"""
        print(f"Adding {len(texts)} documents to vector store...")

        if embeddings is None:
            embeddings = []
            for text in texts:
                embedding = self.get_embedding(text)
                embeddings.append(embedding)

        embeddings = np.array(embeddings).astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to index
        try:
            self.index.add(embeddings)
        except Exception as e:
            print(f"Error adding to index: {str(e)}")
            return  # Exit if adding to index fails

        self.texts.extend(texts)
        self.metadata.extend(metadatas)

        # Save automatically
        self.save_index()
        print(f"✅ Successfully added documents. Total documents: {len(self.texts)}")

    
    def search(self, query, k=3, score_threshold=0.5):
        """Search for similar documents"""
        if len(self.texts) == 0:
            return []
        
        query_embedding = self.get_embedding(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, min(k, len(self.texts)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= score_threshold:  # Valid result with good score
                results.append({
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx],
                    'score': float(score)
                })
        
        return results
    
    def save_index(self):
        """Save the vector store"""
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump({'texts': self.texts, 'metadata': self.metadata}, f)
    
    def load_index(self):
        """Load the vector store"""
        self.index = faiss.read_index(self.index_file)
        with open(self.metadata_file, 'rb') as f:
            data = pickle.load(f)
            self.texts = data['texts']
            self.metadata = data['metadata']
        print(f"Loaded vector store with {len(self.texts)} documents")
    
    def clear(self):
        """Clear all documents"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.texts = []
        self.metadata = []
        self.save_index()
        print("Vector store cleared")
    
    def get_stats(self):
        """Get statistics about the vector store"""
        return {
            'total_documents': len(self.texts),
            'index_size': self.index.ntotal,
            'dimension': self.dimension
        }