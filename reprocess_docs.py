from document_processor import DocumentProcessor

processor = DocumentProcessor()
processor.clear_all_documents()  # Clear existing data
processor.add_sample_documents()  # Reprocess all samples

stats = processor.get_vector_store_stats()
print(f"✅ Success! Processed {stats['total_documents']} document chunks")
