import os
import mimetypes
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from local_vector_store import LocalVectorStore
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

class DocumentProcessor:
    def __init__(self):
        # Initialize local vector store
        print("Initializing local vector store...")
        self.vector_store = LocalVectorStore()
        print("Vector store ready!")

        print("Loading SentenceTransformer model...")
        self.embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        print("Embedding model loaded.")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def process_text_file(self, file_path, document_id):
        """Process a text file and add to vector database"""
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Split into chunks
            chunks = self.text_splitter.split_text(content)
            print(f"Split document into {len(chunks)} chunks")
            
            # Prepare data for vector store
            texts = chunks
            metadatas = []
            for i, chunk in enumerate(chunks):
                metadatas.append({
                    "text": chunk,
                    "document_id": document_id,
                    "chunk_index": i,
                    "source": file_path
                })
            
            # Embed each chunk
            embeddings = self.embedding_model.encode(texts)

            # Add to vector store
            try:
                self.vector_store.add_documents(texts, metadatas, embeddings=embeddings)
            except Exception as e:
                print(f"Error adding documents to vector store: {str(e)}")
                return False

            print(f"Successfully processed {file_path}")
            return True
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return False

    def detect_file_type(self, file_path):
        """Detect file type using multiple methods"""
        # Try MIME type detection
        mime_type, _ = mimetypes.guess_type(file_path)
        
        # Fallback to file extension
        extension = Path(file_path).suffix.lower()
        
        return mime_type, extension

    def process_any_file(self, file_path, document_id):
        """Process any supported file type"""
        mime_type, extension = self.detect_file_type(file_path)
    
        if extension in ['.txt']:
            return self.process_text_file(file_path, document_id)
        elif extension in ['.pdf']:
            return self.process_pdf_file(file_path, document_id)
        elif extension in ['.docx']:
            return self.process_docx_file(file_path, document_id)
        elif extension in ['.csv']:
            return self.process_csv_file(file_path, document_id)
        else:
            print(f"Unsupported file type: {extension}")
            return False

    def get_chunking_strategy(self, document_type):
        """Get optimal chunking strategy based on document type"""
        strategies = {
            'policy': RecursiveCharacterTextSplitter(
                chunk_size=1500, chunk_overlap=300
            ),
            'handbook': RecursiveCharacterTextSplitter(
                chunk_size=2000, chunk_overlap=400
            ),
            'csv_data': RecursiveCharacterTextSplitter(
                chunk_size=800, chunk_overlap=100
            ),
            'default': RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
        }
        return strategies.get(document_type, strategies['default'])
    
    def add_sample_documents(self):
        """Add some sample company documents"""
        sample_docs = {
            "vacation_policy": """
            Company Vacation Policy
            
            All full-time employees are entitled to 15 days of paid vacation per year.
            Vacation days accrue at a rate of 1.25 days per month.
            Employees must request vacation at least 2 weeks in advance.
            Vacation requests should be submitted through the HR portal.
            Unused vacation days cannot be carried over to the next year.
            Maximum vacation that can be taken at once is 10 consecutive days.
            Part-time employees receive prorated vacation based on hours worked.
            """,
            
            "expense_policy": """
            Expense Reimbursement Policy
            
            Employees can be reimbursed for business-related expenses.
            All expenses must be submitted within 30 days with receipts.
            Meals are reimbursed up to $50 per day during business travel.
            Transportation costs for business trips are fully covered.
            Hotel accommodation is covered up to $200 per night.
            Submit expense reports through the finance portal at finance.company.com.
            Approval from manager required for expenses over $500.
            Personal expenses will not be reimbursed under any circumstances.
            """,
            
            "it_support": """
            IT Support Guidelines
            
            For technical issues, contact IT support at it-help@company.com
            For urgent issues, call the IT hotline: (555) 123-4567
            Common issues can be resolved through the self-service portal.
            New equipment requests should be submitted through IT portal.
            Software installation requires IT approval for security reasons.
            Password resets can be done through the company portal.
            VPN access is required for remote work - contact IT for setup.
            Regular security training is mandatory for all employees.
            """,
            
            "remote_work": """
            Remote Work Policy
            
            Employees can work remotely up to 3 days per week.
            Remote work must be pre-approved by direct manager.
            All remote workers must have reliable internet connection.
            Company will provide necessary equipment for home office setup.
            Remote workers must be available during core hours 10am-3pm EST.
            Monthly in-person team meetings are required.
            Productivity metrics will be tracked for remote workers.
            Remote work privileges can be revoked for performance issues.
            """
        }
        
        # Create sample_docs folder if it doesn't exist
        os.makedirs("sample_docs", exist_ok=True)
        
        # Save and process each document
        for doc_id, content in sample_docs.items():
            file_path = f"sample_docs/{doc_id}.txt"
            with open(file_path, 'w') as f:
                f.write(content)
            
            self.process_text_file(file_path, doc_id)
        
        # Print statistics
        stats = self.vector_store.get_stats()
        print(f"Vector store now contains {stats['total_documents']} document chunks")
    
    def get_vector_store_stats(self):
        """Get vector store statistics"""
        return self.vector_store.get_stats()
    
    def clear_all_documents(self):
        """Clear all documents from vector store"""
        self.vector_store.clear()
