from document_processor import DocumentProcessor
from query_engine import QueryEngine

# Test document processing
print("Testing document processor...")
processor = DocumentProcessor()
processor.add_sample_documents()
print("Sample documents added!")

# Test query engine
print("\nTesting query engine...")
engine = QueryEngine()
answer = engine.ask_question("What's our vacation policy?", use_advanced=False)
print(f"Answer: {answer}")

# Test advanced mode (optional)
print("\nTesting advanced mode...")
advanced_answer = engine.ask_question("How do I submit expenses?", use_advanced=True)
print(f"Advanced Answer: {advanced_answer}")