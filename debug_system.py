# debug_system.py - Run this to diagnose issues

from document_processor import DocumentProcessor
from query_engine import QueryEngine
import os

def debug_document_processing():
    """Debug document processing issues"""
    print("🔍 DEBUGGING DOCUMENT PROCESSING")
    print("=" * 50)
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Check if vector store exists
    stats = processor.get_vector_store_stats()
    print(f"📊 Current vector store stats: {stats}")
    
    if stats['total_documents'] == 0:
        print("❌ No documents in vector store!")
        print("💡 Solution: Upload and process documents first")
        return False
    else:
        print(f"✅ Found {stats['total_documents']} documents in vector store")
        return True

def debug_search_functionality():
    """Debug search functionality"""
    print("\n🔍 DEBUGGING SEARCH FUNCTIONALITY")
    print("=" * 50)
    
    engine = QueryEngine()
    
    # Test search with sample questions
    test_questions = [
        "How many vacation days do employees get?",
        "What are the work hours?",
        "What is the dress code?",
        "When are performance reviews?",
        "How many sick days?",
        "What holidays does the company observe?"
    ]
    
    print("Testing search results for sample questions:")
    for question in test_questions:
        print(f"\nQ: {question}")
        chunks = engine.search_documents(question)
        print(f"Found {len(chunks)} relevant chunks")
        
        if chunks:
            print(f"Best match score: {chunks[0]['score']:.3f}")
            print(f"Content preview: {chunks[0]['text'][:100]}...")
        else:
            print("❌ No relevant chunks found!")
    
    return len(chunks) > 0

def debug_answer_generation():
    """Debug answer generation"""
    print("\n🔍 DEBUGGING ANSWER GENERATION")
    print("=" * 50)
    
    engine = QueryEngine()
    
    # Test with your specific question
    test_question = "How many vacation days do employees get?"
    print(f"Testing question: {test_question}")
    
    # Get search results
    chunks = engine.search_documents(test_question)
    print(f"Search found {len(chunks)} chunks")
    
    if chunks:
        print("\nChunk details:")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: Score={chunk['score']:.3f}")
            print(f"Content: {chunk['text'][:200]}...")
            print("-" * 30)
    
    # Generate answer
    answer = engine.ask_question(test_question, use_advanced=False)
    print(f"\nGenerated answer: {answer}")
    
    return answer

def debug_file_processing():
    """Debug file upload and processing"""
    print("\n🔍 DEBUGGING FILE PROCESSING")
    print("=" * 50)
    
    # Check uploads folder
    if os.path.exists("uploads"):
        files = os.listdir("uploads")
        print(f"Files in uploads folder: {files}")
    else:
        print("❌ No uploads folder found")
        print("💡 Solution: Create uploads folder and upload files")
    
    # Check sample_docs folder
    if os.path.exists("sample_docs"):
        files = os.listdir("sample_docs")
        print(f"Files in sample_docs folder: {files}")
    else:
        print("❌ No sample_docs folder found")
        print("💡 Solution: Run processor.add_sample_documents()")
    
    # Check vector_store folder
    if os.path.exists("vector_store"):
        files = os.listdir("vector_store")
        print(f"Files in vector_store folder: {files}")
        
        # Check if index files exist
        if "docs.index" in files and "metadata.pkl" in files:
            print("✅ Vector store files found")
        else:
            print("❌ Vector store files missing")
            print("💡 Solution: Process documents to create vector store")
    else:
        print("❌ No vector_store folder found")

def test_with_your_document():
    """Test with the uploaded TechCorp handbook"""
    print("\n🔍 TESTING WITH YOUR DOCUMENT")
    print("=" * 50)
    
    # Your document content
    document_content = """TechCorp Employee Handbook 2024

WELCOME TO TECHCORP
Welcome to TechCorp! This handbook contains important information about company policies, benefits, and procedures.

WORK HOURS AND TIME OFF
Standard work hours are Monday through Friday, 9:00 AM to 5:30 PM with a one-hour lunch break.
Flexible work arrangements are available with manager approval.
Remote work is permitted up to 3 days per week for eligible employees.

VACATION POLICY
- Full-time employees: 20 vacation days per year
- Part-time employees: Pro-rated based on hours worked
- Vacation must be requested 2 weeks in advance through the HR portal
- Maximum carryover: 5 days to the following year
- Vacation payout available upon termination

SICK LEAVE
All employees receive 10 sick days per year.
Sick leave can be used for personal illness or caring for family members.
Doctor's note required for absences longer than 3 consecutive days.

HOLIDAYS
TechCorp observes the following holidays:
- New Year's Day
- Martin Luther King Jr. Day
- Presidents' Day
- Memorial Day
- Independence Day
- Labor Day
- Thanksgiving (2 days)
- Christmas Day
- 2 floating holidays of employee's choice

DRESS CODE
Business casual attire is expected in the office.
Jeans are permitted on Fridays.
Client-facing roles may require business professional attire.
Remote workers should dress appropriately for video calls.

PERFORMANCE REVIEWS
Annual performance reviews are conducted each January.
Mid-year check-ins occur in July.
Goal setting and development planning are part of the review process."""
    
    # Save the document
    os.makedirs("sample_docs", exist_ok=True)
    with open("sample_docs/techcorp_handbook.txt", "w") as f:
        f.write(document_content)
    
    # Process it
    print("Processing TechCorp handbook...")
    processor = DocumentProcessor()
    success = processor.process_text_file("sample_docs/techcorp_handbook.txt", "techcorp_handbook")
    
    if success:
        print("✅ Document processed successfully!")
        
        # Test questions specific to your document
        engine = QueryEngine()
        test_questions = [
            "How many vacation days do full-time employees get?",
            "What are the standard work hours?",
            "How many sick days do employees receive?",
            "What holidays does TechCorp observe?",
            "What is the dress code policy?",
            "When are performance reviews conducted?",
            "How many days per week can employees work remotely?"
        ]
        
        print("\nTesting questions with your document:")
        for question in test_questions:
            print(f"\n🔵 Q: {question}")
            answer = engine.ask_question(question, use_advanced=False)
            print(f"🔸 A: {answer}")
            print("-" * 60)
    else:
        print("❌ Failed to process document")

def run_full_diagnosis():
    """Run complete system diagnosis"""
    print("🚀 RUNNING FULL SYSTEM DIAGNOSIS")
    print("=" * 60)
    
    # Step 1: Check document processing
    docs_ok = debug_document_processing()
    
    # Step 2: Check file processing
    debug_file_processing()
    
    # Step 3: Process your document
    test_with_your_document()
    
    # Step 4: Test search functionality
    if docs_ok:
        search_ok = debug_search_functionality()
        
        # Step 5: Test answer generation
        if search_ok:
            debug_answer_generation()
    
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSIS COMPLETE")
    print("\nCommon solutions if issues persist:")
    print("1. Delete vector_store folder and reprocess documents")
    print("2. Lower the score threshold in search_documents() method")
    print("3. Check that documents are properly chunked and embedded")
    print("4. Verify question preprocessing is working correctly")

if __name__ == "__main__":
    run_full_diagnosis()