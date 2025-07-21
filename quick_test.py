# quick_test.py
from query_engine import QueryEngine
from document_processor import DocumentProcessor

# Process your document
processor = DocumentProcessor()

# Save your TechCorp content
content = """TechCorp Employee Handbook 2024
VACATION POLICY
- Full-time employees: 20 vacation days per year
- Part-time employees: Pro-rated based on hours worked"""

with open("test_doc.txt", "w") as f:
    f.write(content)

# Process it
processor.process_text_file("test_doc.txt", "test")

# Test query
engine = QueryEngine()
answer = engine.ask_question("How many vacation days do full-time employees get?")
print(f"Answer: {answer}")